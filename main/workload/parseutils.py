import numpy as np
from dateutil.relativedelta import relativedelta
from dateutil import parser
import datetime
import sqlparse
import re

def remove_whitespace(tokens):
	cleaned = []
	for tok in tokens:
		if not str(tok).isspace() and \
				not tok.match(sqlparse.tokens.Error, ("'", '"', "$")) and \
				not (isinstance(tok, sqlparse.sql.Token) and \
					 tok.ttype is sqlparse.tokens.Punctuation):
			cleaned.append(tok)
	return cleaned


def remove_empty(parts):
	cleaned = []
	for p in parts:
		if len(p) > 0:
			cleaned.append(p)
	return cleaned


def remove_wrappers(val, left, right):
	clause = val.lstrip().rstrip()
	idx1 = clause.find(left)
	idx2 = clause.rfind(right)
	if idx1 == 0 and idx2 == len(clause) - 1:
		clause = clause[idx1 + 1:idx2].lstrip().rstrip()
	return clause


def clean_values(col):
	col = col.lstrip().rstrip()
	if 'trim(' == col[:5]:
		col = col[5:-1].lstrip()
	if 'lower(' == col[:6]:
		col = col[6:-1].lower()
	col = remove_cast(col)
	col = remove_trunc(col)
	col = remove_wrappers(col, "'", "'")
	col = remove_wrappers(col, '"', '"')
	if col.isnumeric():
		return int(col)
	else:
		return col


def parse_like(col, op, pattern, dvs, regex=False):
	if not col in dvs:
		return []
	if not regex:
		pattern = clean_values(pattern)
		# Remove white space and unneeded quotes
		p = remove_wrappers(pattern, "(", ")")
		p = remove_wrappers(p, "'", "'")
		p = remove_wrappers(p, '"', '"')
		p = p.replace("*", ".*").replace("%", ".*")
	else:
		p = pattern.replace(")(", "|")
		p = remove_wrappers(p, "'", "'")
		p = remove_wrappers(p, '"', '"')
	match = []
	no_match = []
	for v in dvs[col]:
		if re.search(p, v, re.IGNORECASE) is not None:
			match.append(v)
		else:
			no_match.append(v)
	# print(col, op, pattern)
	# print(match, no_match)
	if 'not' in op:
		return no_match
	else:
		return match


def parse_excludes(col, excludes, dvs):
	if not col in dvs:
		return []
	remains = list(set(dvs[col]).difference(excludes))
	return remains


def ts_interval_tokens(tok):
	return isinstance(tok, sqlparse.sql.Identifier) or \
		   tok.ttype == sqlparse.tokens.Literal.Number.Integer or \
		   'cast' in str(tok) or \
		   str(tok) in ['+', '-', 'interval', 'year',
						'day', 'days', 'week', 'weeks', 'years',
						'month', 'months', 'hour', 'hours']


def is_keyword(tok):
	return isinstance(tok, sqlparse.sql.Token) and \
		tok.ttype is sqlparse.tokens.Token.Keyword


def is_comparison(tok):
	return isinstance(tok, sqlparse.sql.Token) and \
		tok.ttype is sqlparse.tokens.Token.Operator.Comparison or \
		is_keyword(tok) and str(tok) in ['in']


def get_relative_delta(val):
	tmp = val.split("interval")
	delta_str = tmp[1].lstrip().rstrip().split()
	n = int(delta_str[0])
	if not delta_str[0][0] == '-' and not '+' in tmp[0]:
		n *= -1
	if 'hour' in delta_str[1]:
		delta = relativedelta(hours=n)
	elif 'day' in delta_str[1]:
		delta = relativedelta(days=n)
	elif 'week' in delta_str[1]:
		delta = relativedelta(weeks=n)
	elif 'month' in delta_str[1]:
		delta = relativedelta(months=n)
	elif 'year' in delta_str[1]:
		delta = relativedelta(years=n)
	else:
		delta = None
	return delta


def parse_timestamp(col, val, ts):
	val = remove_cast(val)
	val = remove_trunc(val, ts)
	if "null" in val:
		return "nan"
	elif "unix_timestamp(date_trunc('month', now()))" in val:
		return ts.truncate('month')
	elif "date_sub(" in val or "date_add(" in val:
		idx1 = val.find(",")
		if "date_sub(" in val:
			idx3 = val.find("date_sub(")
		else:
			idx3 = val.find("date_add(")
		t0 = parse_timestamp(col, val[idx3+9:idx1].strip(), ts)
		if "interval" in val:
			delta = get_relative_delta(val[idx1+1:])
		else:
			idx2 = val.find(")", idx1+1)
			n = int(val[idx1+1:idx2].strip())
			if "date_sub(" in val:
				n *= -1
			delta = relativedelta(days=n)
		return t0 + delta
	elif "months_add(" in val:
		idx1 = val.find(",")
		idx3 = val.find("months_add(")
		t0 = parse_timestamp(col, val[idx3 + 11:idx1].strip(), ts)
		if "interval" in val:
			delta = get_relative_delta(val[idx1 + 1:])
		else:
			idx2 = val.find(")", idx1 + 1)
			n = int(val[idx1 + 1:idx2].strip())
			delta = relativedelta(months=n)
		return t0 + delta
	elif "unix_timestamp()" in val:
		return ts
	elif re.search('from_unixtime\(\s*unix_timestamp\(', val) is not None:
		idx1 = re.search('from_unixtime\(\s*unix_timestamp\(', val).span()[1] + 1
		idx2 = val.find(",", idx1)
		t = val[idx1:idx2]
		# Remove PM from time string; otherwise parser fails
		idx4 = t.find("pm")
		if idx4 > -1:
			t = t[:idx4].strip()
		return parse_timestamp(col, t, ts)
	elif "unix_timestamp(" in val:
		idx1 = val.find("unix_timestamp(")
		idx2 = val.rfind(")")
		return parse_timestamp(col, val[idx1+15:idx2], ts)
	elif "now(" in val or "utc_timestamp(" in val or "now" in val or 'utc_timestamp' in val:
		return ts
	elif col in {"pa__arrival_period", "pa__arrival_day"} and val.isdecimal():
		return datetime.datetime.fromtimestamp(int(val))
	elif col == "pa__arrival_period" and "(" in val and ")" in val:
		idx1 = val.find("(")
		idx2 = val.rfind(")")
		tmp = val[idx1 + 1:idx2].split(",")
		return [clean_values(t) for t in tmp]
	else:
		if val[0] == '\"' or val[0] == "\'":
			val = val[1:]
		if val[-1] == '\"' or val[-1] == "\'":
			val = val[:-1]
		if val[-1] == "z":
			return datetime.datetime.strptime(val, "%Y-%m-%dt%H:%M:%Sz")
		else:
			return parser.parse(val)


def format_ts(col, t):
	if isinstance(t, str) or isinstance(t, list):
		return t
	# Covert to timestamp
	if col in {"pa__arrival_period", "pa__arrival_day"}:
		return int(datetime.datetime.timestamp(t))
	# Convert to datetime string
	else:
		return t.strftime("%Y-%m-%d %H:%M:%S")


def remove_cast(val):
	while "cast(" in val:
		idx1 = val.find("(", 4)
		idx2 = val.rfind(" as ")
		val = val[idx1 + 1:idx2]
	return val


def remove_trunc(val, ts=None):
	if "trunc(" in val:
		idx1 = val.find("(", 5)
		idx2 = val.find(",", idx1+1)
		if "now()" in  val and 'mm' in val:
			val = ts.truncate("month")
		else:
			val = val[idx1 + 1:idx2]
	return val


def is_time_col(col):
	return col in {"pa__arrival_day", "start_time", "end_time", "pa__arrival_period",
		"pa__processed_ts", "pa__arrival_ts", "envelope_ts"}


def parse_value(col, val, ts):
	if val.lower() == "true":
		return True
	elif "none" in val:
		return np.nan
	elif col == "size_in_bytes" and '*' in val:
		# Evaluate expression like 1*1024*1024
		p = 1
		tmp = val.split("*")
		for n in tmp:
			p *= int(n)
		return p
	elif val.isnumeric():
		return int(val)
	elif "interval" in val and not ("date_sub(" in val or "date_add(" in val):
		delta = get_relative_delta(val)
		val = val.split("interval")[0]
		# Remove the interval part from string
		if '+' in val:
			idx = val.rfind('+')
			val = val[:idx].strip()
		else:
			idx = val.rfind('-')
			val = val[:idx].strip()
		t0 = parse_timestamp(col, val, ts)
		return format_ts(col, t0 + delta)
	elif is_time_col(col):
		t0 = parse_timestamp(col, val, ts)
		return format_ts(col, t0)
	# elif "cast(" in val and "trunc(" in val and "now()" in val:
	# 	if 'mm' in val:
	# 		t0 = ts.truncate('month')
	# 	elif 'dd' in val:
	# 		t0 = ts.truncate('day')
	# 	if 'bigint' in val or 'timestamp' in val:
	# 		t0 = t0.timestamp()
	# 	return format_ts(col, t0)
	elif "cast(" in val and " timestamp" in val:
		t0 = parser.parse(remove_cast(val))
		t0 = t0.timestamp()
		return format_ts(col, t0)
	elif "(" in val and "," in val and ")" in val:
		idx1 = val.find("(")
		idx2 = val.rfind(")")
		tmp = val[idx1+1:idx2].split(",")
		return [clean_values(t) for t in tmp]
	else:
		if "(" in val and ")" in val:
			idx1 = val.find("(")
			idx2 = val.rfind(")")
			val = val[idx1+1:idx2]
			return [clean_values(val)]
		else:
			return clean_values(val)


def parse_column_name(col, expr):
	if isinstance(col, int):
		return "_", expr
	if "trunc(" in col:
		col = remove_trunc(col)
	if "to_date(" in col:
		col = col[8:-1]
	elif "to_timestamp(" in col:
		col = col[13:-1]
	elif "from_unixtime(unix_timestamp" in col:
		idx1 = col.find("(", 28)
		idx2 = col.find(")", idx1 + 1)
		col = col[idx1 + 1:idx2]
	elif "cast(" in col:
		idx1 = col.find("(", 4)
		idx2 = col.find(" as", idx1 + 1)
		if "interval" in col:
			idx3 = col.find(")", idx2 + 1)
			interval_str = col[idx3 + 1:]
			idx4 = interval_str.find("interval")
			if "-" in interval_str[:idx4]:
				expr[2] = expr[2] + "+" + interval_str[idx4:]
			else:
				expr[2] = expr[2] + "-" + interval_str[idx4:]
		col = col[idx1 + 1:idx2]
	elif " not" in col:
		col = col.split()[0]
		expr[1] = "not " + expr[1]
	if "." in col:
		col = col.split(".")[1]
	return col, expr