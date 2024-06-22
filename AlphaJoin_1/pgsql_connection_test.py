import psycopg2

conn = psycopg2.connect(database="imdb", user="postgres", password="postgres", host="localhost", port="5432")
cur = conn.cursor()
# cur.execute("SET join_collapse_limit TO 1;")
cur.execute("explain "+'''SELECT MIN(n.name) AS voicing_actress,
       MIN(t.title) AS jap_engl_voiced_movie
FROM company_name AS cn
JOIN movie_info AS mi ON cn.country_code = 'us' AND mi.info IS NOT NULL
JOIN title AS t ON mi.movie_id = t.id AND t.production_year > 2006
JOIN movie_companies AS mc ON t.id = mc.movie_id AND cn.id = mc.company_id
JOIN cast_info AS ci ON t.id = ci.movie_id
JOIN role_type AS rt ON ci.role_id = rt.id AND rt.role = 'actress'
JOIN char_name AS chn ON ci.person_role_id = chn.id
JOIN name AS n ON ci.person_id = n.id AND n.gender = 'f' AND n.name LIKE '%Atalya%'
JOIN aka_name AS an ON ci.person_id = an.person_id
JOIN info_type AS it ON mi.info_type_id = it.id AND it.info = 'release dates'
WHERE (mi.info LIKE 'Japan:% 200%' OR mi.info LIKE 'USA:% 200%')
  AND ci.note IN ('(executive producer)', '(uncredited)', '(voice)', '(as Atalya) (credit only)')
GROUP BY rt.id;


''')
rows = cur.fetchall()
# print(rows)
for row in rows:
    print(row)


conn.close()
cur.close()

#('Planning Time: 8.553 ms',)
#('Execution Time: 951.307 ms',)