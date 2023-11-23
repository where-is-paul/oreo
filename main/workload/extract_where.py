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


def get_where(tokens, tbl):
	for i in range(len(tokens)-1):
		if tbl in str(tokens[i]) and 'where' in str(tokens[i+1]).lower():
			return str(tokens[i+1])
	return None


def format_where(where):
	where = where.replace('\n', ' ')
	return " ".join(where.split())


def format_query(q):
	# Remove comment
	idx = q.find("--")
	while idx != -1:
		end_idx = q.find("\n", idx)
		q = q[:idx] + q[end_idx+1:]
		idx = q.find("--")
	# Remove leading and trailing quotes
	idx1 = q.find('"')
	if idx1 != -1 and (idx1 == 0 or q[:idx1].isspace()):
		q = q[idx1+1:]
		idx2 = q.rfind('"')
		if idx2 != -1 and q[idx2+1:].isspace():
			q = q[:idx2]
	return q.replace('`', '').lower()


def is_with(token):
	return token.ttype == sqlparse.tokens.CTE and "with" in str(token)


def get_with_inner_from_tokens(tokens, tbl):
	res = []
	if is_with(tokens[0]):
		q = str(tokens[1])
		indices = []
		# WITH view1 AS (), view2 AS(), ...
		# sql parse does not deal with this case
		p = re.compile(r'as[\n\r\s]+\(')
		iterator = p.finditer(q)
		for match in iterator:
			indices.append(match.span()[0])
		if len(indices) == 1:
			return [q]
		for i in range(len(indices) - 1):
			substr = q[indices[i]:indices[i+1]]
			end_idx = substr.rfind(")")
			if tbl in substr[:end_idx+1]:
				res.append(substr[:end_idx+1])
		return res
		
	for i in range(len(tokens) - 1):
		if "from" in str(tokens[i]) and is_with(tokens[i+1]):
			q = str(tokens[i+1])
			idx1 = q.find("(")
			idx2 = q.rfind(")")
			stmt = q[idx1+1:idx2]
			parsed = sqlparse.parse(stmt)[0]
			new_tokens = remove_whitespace(parsed.tokens)
			return get_with_inner_from_tokens(new_tokens, tbl)
	return []


def strip_with(statements, tbl):
	cleaned = []
	for stmt in statements:
		if not 'with' in stmt:
			cleaned.append(stmt)
			continue
		parsed = sqlparse.parse(stmt)[0]
		tokens = remove_whitespace(parsed.tokens)
		qs = get_with_inner_from_tokens(tokens, tbl)
		for q in qs:
			idx1 = q.find("(")
			idx2 = q.rfind(")")
			inner = q[idx1+1:idx2]
			new = strip_with(sqlparse.split(inner), tbl)
			cleaned.extend(new)
	return cleaned


def _parse_subquey(query):
	idx1 = query.find("(")
	idx2 = query.rfind(")")
	inner = query[idx1+1:idx2]
	new_tokens = remove_whitespace(sqlparse.parse(inner)[0].tokens)
	return new_tokens


def extract_subquery(tokens, tbl):
	results = remove_whitespace(tokens)
	for tok in tokens:
		q = str(tok)
		if tbl in q:
			new_tokens = _parse_subquey(q)
			# Has inner queries
			if len(new_tokens) > 1:
				results = extract_subquery(new_tokens, tbl)
				break
	return results 


def strip_query(row, tbl):
	vals = row.split("|")
	ts = vals[0] + "|" + vals[1]
	q = ''.join(vals[6:])
	results = []
	if tbl in q:
		raw = format_query(q)
		statements = strip_with(sqlparse.split(raw), tbl)
		for stmt in statements:
			if len(stmt) == 0 or tbl not in stmt:
				continue
			parsed = sqlparse.parse(stmt)[0]
			tokens = extract_subquery(parsed.tokens, tbl)
			if len(tokens) > 1:
				where = get_where(tokens, tbl)
				if where is not None:
					results.append((ts, format_where(where)))
	return results


if __name__ == "__main__":
	tables = ["history.bundle"]
	for tbl in tables:
		fname = tbl.split(".")[1]
		f = open('raw/%s.csv' % fname, 'r')
		g = open('where/%s.txt' % fname, 'w')
		lines = []
		for cnt, line in enumerate(f.readlines()):
			if cnt == 0:
				continue
			if line[:5] in ['2021-', '2020-', '2019-'] and len(lines) > 0:
				row = ''.join(lines)
				ans = strip_query(row, tbl)
				for (ts, where) in ans:
					g.write("%s|%s\n" % (ts, where))
				lines = []
			lines.append(line)
			if cnt % 10000 == 0:
				print(cnt)
		# Last query 
		row = ''.join(lines)
		ans = strip_query(row, tbl)
		for (ts, where) in ans:
			g.write("%s|%s\n" % (ts, where))
		f.close()
		g.close()
