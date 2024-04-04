import psycopg2

conn = psycopg2.connect(database="imdb", user="postgres", password="postgres", host="localhost", port="5432")
cur = conn.cursor()

cur.execute("select * from company_type")
rows = cur.fetchall()
print(rows)
for row in rows:
    print(row)


conn.close()
cur.close()