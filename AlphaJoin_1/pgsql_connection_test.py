import psycopg2

conn = psycopg2.connect(database="imdb", user="postgres", password="postgres", host="localhost", port="5432")
cur = conn.cursor()
cur.execute("SET join_collapse_limit TO 1;")
cur.execute("explain "+'''SELECT MIN(chn.name) AS uncredited_voiced_character,
       MIN(t.title) AS russian_movie
FROM (company_name AS cn
JOIN movie_companies AS mc ON cn.id = mc.company_id)
JOIN (role_type AS rt
JOIN cast_info AS ci ON ci.role_id = rt.id
     AND ci.note LIKE '%(voice)%'
     AND ci.note LIKE '%(uncredited)%')
ON ci.movie_id = mc.movie_id
JOIN char_name AS chn ON chn.id = ci.person_role_id
JOIN title AS t ON t.id = ci.movie_id
LEFT JOIN company_type AS ct ON mc.company_type_id = ct.id
WHERE cn.country_code = '[ru]'
AND t.production_year > 2005;
''')
rows = cur.fetchall()
# print(rows)
for row in rows:
    print(row)


conn.close()
cur.close()

#('Planning Time: 8.553 ms',)
#('Execution Time: 951.307 ms',)