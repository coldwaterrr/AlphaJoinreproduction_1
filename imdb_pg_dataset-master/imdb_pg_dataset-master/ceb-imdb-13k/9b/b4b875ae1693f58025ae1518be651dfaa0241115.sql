SELECT mi1.info, n.name, COUNT(*)
FROM title as t,
kind_type as kt,
movie_info as mi1,
info_type as it1,
cast_info as ci,
role_type as rt,
name as n,
info_type as it2,
person_info as pi
WHERE
t.id = ci.movie_id
AND t.id = mi1.movie_id
AND mi1.info_type_id = it1.id
AND t.kind_id = kt.id
AND ci.person_id = n.id
AND ci.movie_id = mi1.movie_id
AND ci.role_id = rt.id
AND n.id = pi.person_id
AND pi.info_type_id = it2.id
AND (it1.id IN ('3','6','8'))
AND (it2.id IN ('35'))
AND (mi1.info IN ('Action','Adventure','Animation','Australia','Austria','Drama','East Germany','Family','Germany','Japan','Mexico','Reality-TV','Spain','USA','War'))
AND (n.name ILIKE '%st%')
AND (kt.kind IN ('tv movie','tv series','video game'))
AND (rt.role IN ('cinematographer','composer','editor','production designer','writer'))
AND (t.production_year <= 2015)
AND (t.production_year >= 1925)
GROUP BY mi1.info, n.name