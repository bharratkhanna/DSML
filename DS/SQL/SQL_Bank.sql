-- Day 06

CREATE DATABASE bank;
USE bank;

SELECT * FROM account;
SELECT * FROM transaction;
SELECT * FROM customer;

SHOW TABLES;  

-- DISTINCT - Show unique values only
SELECT DISTINCT account_type FROM account;    

-- Order By - For Sorting
SELECT transaction_id,transaction_amount 
FROM transaction
ORDER BY transaction_amount DESC;         

-- COUNT
SELECT COUNT(DISTINCT account_type) 
FROM account;

SELECT COUNT(*) FROM customer;

-- LIMIT
SELECT transaction_id, transaction_amount 
FROM transaction
transaction_amount LIMIT 5;

SELECT transaction_amount FROM transaction
ORDER BY transaction_amount DESC
LIMIT 1;

-- MAX
SELECT MAX(transaction_amount) 
FROM transaction;

SELECT transaction_amount FROM transaction
WHERE transaction_amount < 50000
ORDER BY transaction_amount DESC
LIMIT 1;

SELECT transaction_amount FROM transaction
ORDER BY transaction_amount DESC
LIMIT 1,1; -- First is start row no, Second is How many records to show after that?

SELECT DISTINCT transaction_amount FROM transaction
ORDER BY transaction_amount DESC
LIMIT 2,1;

-- AVG
SELECT AVG(transaction_amount) AS avg_transaction 
FROM transaction;

SELECT customer_id, MAX(balance_amount) 
FROM account;

# -------------------------------------------------------------------------------------|

-- Day 07

USE bank;
