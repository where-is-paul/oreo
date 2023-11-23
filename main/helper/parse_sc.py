import pandas as pd
import json
from ..workload.parseutils import *
from ..utils.predicates import *
import pickle


def parse_and_build_predicate(cfg, expr, ts):
	if len(expr) < 3:
		print("len < 3", expr)
		return None
	# Special case: materialize subquery
	if "cpbu_reporting.vcenter_customers_monthly_cpbutr_" in expr[2]:
		yy = ts.year
		mm = ts.month
		expr[2] = "%d-%d-01" % (yy, mm)
	predicate = {}
	col = expr[0]
	op = str(expr[1])
	# This column has too many distinct values to manually expand like
	# Ignore this predicate
	if col == "pa__collector_instance_id" and 'like' in op:
		return None
	# Ignore null check for now
	if "null" in expr[2]:
		return None
	# Clean up column name
	if not col in all_cols:
		col, expr = parse_column_name(col, expr)
	# Not supported: (pa__arrival_day / 86400) % 5 == 1
	if not col in all_cols:
		return None
	# Ignore 0 selectivity time predicates
	#if ts < datetime.datetime(2021, 6, 1) and col in time_cols:
	#	return None

	# Parse operands
	try:
		# Special case
		if 'like' in op and col == 'query_status':
			predicate[col] = parse_excludes(col, ['ok'], dvs)
		elif 'rlike' in op:
			predicate[col] = parse_like(col, op, expr[2], dvs, True)
		elif 'like' in op or 'ilike' in op:
			predicate[col] = parse_like(col, op, expr[2], dvs)
		elif op == "=":
			predicate[col] = [parse_value(col, expr[2], ts)]
		elif op in [">", ">="]:
			predicate[col] = [parse_value(col, expr[2], ts), np.nan]
		elif op in ["<", "<="]:
			predicate[col] = [np.nan, parse_value(col, expr[2], ts)]
		elif 'between' in op:
			predicate[col] = [parse_value(col, expr[2], ts), parse_value(col, expr[3], ts)]
		elif op == "not in" or op == '!=':
			predicate[col] = parse_excludes(col, expr[2:], dvs)
		elif op == "in" or op == 'is':
			# Special case: materialize subquery
			if 'from history.tdm_project_info' in expr[2]:
				predicate[col] = ["tdm.2_5_0", "tdm.3_0_0", "tdm.2_1_1", "tdm.2_0_0",
					"tdm.2_1_8", "tdm.2_1_9", "tdm.2_1_4", "tdm.2_1_0", "tdm.2_2_1",
					"tdm.2_1_2", "tdm.2_1_7"]
			elif "select collector_id from views.vmc_collectors" in expr[2]:
				predicate[col] = parse_like(col, 'rlike', ".*vmc.*", dvs, True)
			else:
				predicate[col] = parse_value(col, expr[2], ts)
				if col in dvs:
					predicate[col] = list(set(predicate[col]).intersection(dvs[col]))
	except:
		print("parse error", expr, col, op)
		return None
	# No regex match for like operators
	if len(predicate[col]) == 0:
		print("No match", col, expr, ts)
		return None
	return Expr(cfg, predicate)


def update_stack(stack, new):
	if new is None:
		return stack
	if len(stack) > 0 and stack[-1] in [LogicalOp.AND, LogicalOp.OR]:
		op = stack.pop()
		left = stack.pop()
		new_clause = Predicate(left, new, op)
		stack.append(new_clause)
	else:
		stack.append(new)
	return stack

def preprocess(query, ts):
	query = query.lower().replace('\.', '.')
	# Edge case
	s = "false or lower(vendorname) in"
	if s in query:
		idx = query.index(s)
		query = query[:idx] + "lower(vendorname) in" + query[idx + len(s):]
	s = "concat("
	if s in query:
		idx1 = query.index(s)
		idx2 = query.index(")", idx1+len(s))
		vals = query[idx1+len(s):idx2].split(",")
		cleaned = [clean_values(v) for v in vals]
		query = query[:idx1] + "'" + ''.join(cleaned) + "'" + query[idx2+1:]
	if "unix_timestamp(date_trunc('month', now()))" in query:
		query = query.replace("unix_timestamp(date_trunc('month', now()))",
			datetime.datetime(ts.year, ts.month, 1).strftime("%Y-%m-%d %H:%M:%S"))
	if "trunc(now(), 'mm')" in query:
		query = query.replace("trunc(now(), 'mm')",
			datetime.datetime(ts.year, ts.month, 1).strftime("%Y-%m-%d %H:%M:%S"))
	return query

def parse(ts, q):
	query = preprocess(q, ts)
	try:
		p = sqlparse.parse(query)[0]
		tokens = remove_whitespace(p.tokens)
	except:
		print(query)
		tokens = []
	stack = []
	i = 0
	while i < len(tokens):
		tok = tokens[i]
		if 'length(' in str(tok) or 'true or' in str(tok):
			i += 1
			continue
		# Recursively parse expressions inside parenthesis
		if isinstance(tok, sqlparse.sql.Parenthesis):
			inner = remove_wrappers(str(tok), "(", ")")
			clause = parse(ts, inner)
			stack = update_stack(stack, clause)
			i += 1
		# Comparison
		elif isinstance(tok, sqlparse.sql.Comparison):
			j = i + 1
			operator_template = ">=|<=|!=|=|>|<|\s*not like\s+|\s*like\s+|\s*rlike\s+|\s*ilike\s+|\s*in\s+"
			operands = re.split(operator_template, str(tok))
			try:
				operator = re.search(operator_template, str(tok)).group().lstrip().rstrip()
			except:
				if 'like' in str(tok):
					operands = str(tok).split("like")
					operator = 'like'
				elif 'in' in str(tok):
					operands = str(tok).split("in")
					operator = 'in'
				else:
					print(str(tok))
			clause = [clean_values(operands[0]), operator]
			val = [operands[1].lstrip().rstrip()]
			# Trailing: - interval X days
			while j < len(tokens) and ts_interval_tokens(tokens[j]):
				val.append(str(tokens[j]))
				j += 1
			clause.append(" ".join(val))
			expr = parse_and_build_predicate(config, clause, ts)
			stack = update_stack(stack, expr)
			i = j
		# Identifier
		elif isinstance(tok, sqlparse.sql.Identifier):
			clause = [clean_values(str(tok))]
			if i < len(tokens) - 1:
				# rlike is tokenized with identifier
				if ' rlike' in str(tok):
					tmp = str(tok).split()
					clause = [tmp[0], "rlike", str(tokens[i + 1])]
					i += 2
				# like parsed in a function token
				elif isinstance(tokens[i + 1], sqlparse.sql.Function):
					clause.extend(["like", str(tokens[i + 1])[4:]])
					i += 2
				# comparison (<, >, <=, >=, is)
				elif i < len(tokens) - 2 and \
						(is_comparison(tokens[i + 1]) or 'is' in str(tokens[i + 1])):
					clause.extend([str(tokens[i + 1]), str(tokens[i + 2])])
					i += 3
				# not in
				elif i < len(tokens) - 3 and \
						('not' in str(tokens[i + 1]) and 'in' in str(tokens[i + 2])):
					clause.extend(["not in", str(tokens[i + 3])])
					i += 4
				# between ... and ...
				elif i < len(tokens) - 4 and \
						'between' in str(tokens[i + 1]):
					clause.extend(["between", str(tokens[i + 2]), str(tokens[i + 4])])
					i += 5
				# boolean identifier
				elif 'and' in str(tokens[i + 1]) or 'or' in str(tokens[i + 1]):
					clause.extend(['=', "True"])
					i += 1
				else:
					i += 1
			else:
				clause.extend(['=', "True"])
				i += 1
			expr = parse_and_build_predicate(config, clause, ts)
			stack = update_stack(stack, expr)
		# Function
		elif isinstance(tok, sqlparse.sql.Function):
			# function() between ... and ...
			if i < len(tokens) - 4 and \
					'between' in str(tokens[i + 1]):
				clause = [str(tok), 'between',
						str(tokens[i + 2]), str(tokens[i + 4])]
				expr = parse_and_build_predicate(config, clause, ts)
				stack = update_stack(stack, expr)
				i += 5
			else:
				i += 1
				expr = parse_and_build_predicate(config, [str(tok)], ts)
				stack = update_stack(stack, expr)
		# AND, OR
		elif is_keyword(tok):
			if len(stack) > 0 and not stack[-1] in [LogicalOp.AND, LogicalOp.OR]:
				if 'and' in str(tok) and len(stack) > 0:
					stack.append(LogicalOp.AND)
				elif 'or' in str(tok) and len(stack) > 0:
					stack.append(LogicalOp.OR)
			i += 1
		else:
			i += 1

	if len(stack) == 1:
		return stack.pop()
	elif len(stack) > 1:
		return stack[0]
	else:
		return None


def get_dvs():
	# Get distint values
	files = []
	path = "workload/dvs/%s" % ds
	for f in listdir(path):
		if isfile(join(path, f)) and ('csv' in f):
			files.append(f)
	dvs = {}
	for f in files:
		df = pd.read_csv(join(path, f))
		col = f.split('.')[0]
		dvs[col] = list(df[col].fillna("").str.strip().str.lower())
	return dvs


if __name__ == "__main__":
	ds = "analytics__query"
	# Setup
	dvs = get_dvs()
	with open("resources/config/%s.json" % ds, mode="r") as f:
		config = json.load(f)
	all_cols = list(config["cat_cols"])
	all_cols.extend(config["num_cols"])
	all_cols.extend(config["date_cols"])
	#df, _ = load_df(config)
	#df = df.sample(n=min(5000000, len(df))).reset_index(drop=True)
	#N = len(df)
	#print("#rows: %d" % N)

	time_cols = ["pa__arrival_period", "pa__arrival_day", "pa__processed_ts", "pa__arrival_ts"]
	f = open("workload/where/%s.txt" % ds, "r")
	queries = {}
	cnt = 0
	for ln, line in enumerate(f.readlines()):
		# Downsample template
		if "where cast(pa__arrival_day as timestamp) > now() - interval 7 days" in line and \
				np.random.uniform() > 0.05:
			continue
		if "..." in line:
			idx = line.rfind(",")
			line = line[:idx] + ")"
		vals = line.split("|")
		ts = datetime.datetime.strptime(vals[0], '%Y-%m-%d %H:%M:%S')
		if ts < datetime.datetime(2021, 1, 1):
			continue
		# remove where
		query = vals[2].strip()[6:]
		cl = parse(ts, query)
		if cl is not None:
			# try:
			# 	mask = cl.eval(df)
			# 	idx1 = np.nonzero(mask.values)[0]
			# 	if len(idx1) == 0 or len(idx1) == N:
			# 		print("Skipping", len(idx1), cl, ts)
			# 		continue
			# except:
			# 	continue
			queries[str(cnt)] = cl
			cnt += 1
			if cnt % 1000 == 0:
				print(cnt, ln)
				print(cl)
				pickle.dump(queries, open("resources/query/%s-all.p" % ds, "wb"))
	f.close()
	print(len(queries))
	pickle.dump(queries, open("resources/query/%s-all.p" % ds, "wb"))

	# queries = pickle.load(open("resources/query/%s-all.p" % ds, "rb"))
	# sampled = {}
	# N = len(queries)
	# cutoff = int(N * 0.75)
	# s1 = sorted(np.random.choice(cutoff, 5000, replace=False))
	# #s2 = sorted(np.random.choice(range(cutoff, N), 25000, replace=False))
	# s2 = range(cutoff, N)
	# #selected = list(s1)
	# selected = []
	# selected.extend(s2)
	# # selected = list(range(len(queries)))
	# uniq = set()
	# collect_uniq = set()
	# uniq_cols = set()
	# for i, idx in enumerate(selected):
	# 	q = queries[str(idx)]
	# 	sampled[str(i)] = q
	# 	s = str(q)
	# 	if not s in uniq:
	# 		uniq.add(s)
	# 	leaves = q.get_leaves()
	# 	for l in leaves:
	# 		if l.col == "pa__collector_id":
	# 			vals = '|'.join(sorted(l.vals))
	# 			if not vals in collect_uniq:
	# 				collect_uniq.add(vals)
	# 		if not l.col in uniq_cols:
	# 			uniq_cols.add(l.col)
	# print(uniq_cols)
	# print(len(sampled), len(uniq), len(collect_uniq))
	# pickle.dump(sampled, open("resources/query/%s.p" % ds, "wb"))
	#
	#
