SELECT n.name, mi1.info, MIN(t.production_year), MAX(t.production_year)
FROM title as t,
kind_type as kt,
movie_info as mi1,
info_type as it1,
cast_info as ci,
role_type as rt,
name as n
WHERE
t.id = ci.movie_id
AND t.id = mi1.movie_id
AND mi1.info_type_id = it1.id
AND t.kind_id = kt.id
AND ci.person_id = n.id
AND ci.movie_id = mi1.movie_id
AND ci.role_id = rt.id
AND (it1.id IN ('3','8'))
AND (mi1.info IN ('Action','Adult','Adventure','Animation','Comedy','Drama','Family','Fantasy','France','Horror','Music','Sci-Fi','Short','Thriller','UK','USA'))
AND (n.name ILIKE '%lie%')
AND (kt.kind IN ('video game','video movie'))
AND (rt.role IN ('editor','miscellaneous crew','producer','production designer'))
GROUP BY mi1.info, n.name