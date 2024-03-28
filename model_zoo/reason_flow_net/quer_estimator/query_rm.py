def get_query(qtype, reason, kb):
	if not kb :
		a, b = reason[0]['e1_label'].lower(), reason[0]['e2_label'].lower()
		if qtype == 0:
			return ('Qabs', [a], [b])
		r1 = reason[0]['r'].lower()
		if qtype == 1:
			return ('Qars', [a], [r1])
		if qtype == 2:
			return ('Qrbs', [r1], [b])

		c = reason[1]['e2_label'].lower()
		r2 = reason[1]['r'].lower()
		if qtype == 3:
			return ('Qabs', ('Qars', [a], [r1]), [c])
		if qtype == 4:
			return ('Qabs', [a], ('Qrbs', [r2], [c]))
		if qtype == 5:
			return ('Qars', ('Qars', [a], [r1]), [r2])
		if qtype == 6:
			return ('Qrbs', [r1], ('Qrbs', [r2], [c]))
	else:
		a, b = reason[0]['e1_label'].lower(), reason[0]['e2_label'].lower()
		r1 = reason[0]['r'].lower()
		if qtype == 2:
			return ('Qsg', ('Qrbk', [r1], [b]))

		c = reason[1]['e2_label'].lower()
		r2 = reason[1]['r'].lower()
		if qtype == 3:
			return ('Qabk', ('Qars', [a], [r1]), [c])
		if qtype == 4:
			return ('Qabs', [a], ('Qrbk', [r2], [c]))
		if qtype == 5:
			return ('Qark', ('Qars', [a], [r1]), [r2])
		if qtype == 6:
			return ('Qrbs', [r1], ('Qrbk', [r2], [c]))