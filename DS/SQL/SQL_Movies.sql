-- Day 04

CREATE DATABASE IMDB;
USE IMDB;

DROP TABLE movies;

SELECT * FROM movies;
DESCRIBE movies;

-- Operators
-- Mathematical Operators +,-,*,/

SELECT rating,votes FROM movies;
SELECT name Name,rating,votes,rating * votes FROM movies;

-- Alias
SELECT name,actor1,rating*votes AS overall_voting FROM movies;

-- Comparisan Operator >,<,>=,<=,=,!=
SELECT name,rating FROM movies;

-- WHERE
SELECT name,rating FROM movies
WHERE rating > 5;

SELECT name,actor1 FROM movies
WHERE votes > 200;

SELECT name,actor1,director FROM movies
WHERE actor1 = "Aftab";

SELECT name,actor1,director FROM movies
WHERE actor2 = "Aftab";

SELECT name,actor1,director FROM movies
WHERE actor2 != "Aftab";

-- Logical Operator AND,OR,NOT

SELECT name,actor1,director FROM movies
WHERE actor1 = "AJAY" and actor2 = "Akshay";

SELECT name,actor1,director FROM movies
WHERE actor1 = "AJAY" and (actor2 = "AJAY" or actor2 ="Salman");

SELECT name,actor1,director FROM movies
WHERE votes > 200 or actor1 = "Amitabh" or director ="Subhash Ghai";

SELECT name,actor1,director FROM movies
WHERE rating > 6 or actor1 IN ('AJAY','Salman','Amitabh','Aftab');

# ------------------------------------------------------------------------------------

-- Day 05

SELECT * FROM movies;

-- Like : (Used with % or _) When you don know exact word, ALso known as a wild card

SELECT name, actor1 FROM movies
WHERE actor1 LIKE "ay%"; -- or "Ajay%"

	-- ay% -> Letter starts with ay
    -- %ay -> Letter ends with ay
    -- %ay% -> Anywhere in between
    
SELECT name,actor1 FROM movies
WHERE actor1 LIKE '%singh%';

SELECT name,director FROM movies
WHERE director LIKE '%Rajamouli%';

SELECT name,director FROM movies
WHERE director LIKE 'rohit%' and director LIKE '%shetty';

SELECT name,genre,rating FROM movies
WHERE rating > 5 AND genre LIKE '%comedy%';

SELECT name,genre,rating FROM movies
WHERE rating >= 5 AND rating <= 7 AND genre LIKE  '%comedy%';

SELECT name,genre,rating FROM movies
WHERE rating BETWEEN 5 AND 7 AND genre LIKE  '%comedy%';

SELECT name,rating FROM movies
WHERE rating LIKE '5%';

-- _ Searching
SELECT name,actor1 FROM movies
WHERE actor1 LIKE '__y%'; -- First two letters can be anything, 3rd letter wil be y

SELECT votes FROM movies
WHERE votes LIKE '8___0%'; --  "," is also considered

-- DISTINCT - Unique Values
SELECT DISTINCT actor1,name FROM movies;

-- ORDER BY - Ascending Order
SELECT DISTINCT actor1 
FROM movies ORDER BY actor1;

SELECT DISTINCT actor1 
FROM movies ORDER BY actor1 DESC;

# ------------------------------------------------------------------------------------

