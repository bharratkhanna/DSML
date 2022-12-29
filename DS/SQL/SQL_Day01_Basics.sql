 -- This is used for comments
 -- SQL is not a case sensetive language
 -- It is advisable to write keywords/commands in UPPERCASE letter
 -- It is advisable to give proper line break
 
  /*
	Multiple Lines Comment
  */

-- Delete Database
DROP DATABASE DSP; 

-- Database Creation
CREATE DATABASE DSP; -- Data Science Program 

-- View all DATABASES
SHOW DATABASES;

-- Use DATABASE to create,modify and alter data
USE DSP;

-- TABLE CREATION
-- CREATE TABLE <table name> -> to create table name
CREATE TABLE student(
	studentID INT,
    name VARCHAR(50),
    gender VARCHAR(1),
    age INT
    );

-- View Table
SELECT * FROM student;

-- Insert Values
INSERT INTO student(studentID,name,gender,age)
VALUES(101,'Jyoti','F',21);
    
SELECT * FROM student;   

INSERT INTO student(studentID,name,gender,age)
VALUES
		(102,'Ridhi', 'F', 20),
		(103,'Vaibahav', 'M', 21),
        (104, 'Deb', 'M', 20);

-- Delete Table
DROP TABLE student;

SELECT * FROM student;

# ------------------------------------------------------------------------------------

-- Day02

USE DSP;

-- Insert Default Way
INSERT INTO student
	VALUES (106,"Bharat","M",25);
    
SELECT * FROM student;

-- Insert for columns only
INSERT INTO student(name,age)
	VALUES ("Tej",25);


-- DESC describe the table    
DESC student; 
	-- NULL tells whether this columna allow adding Null value or not
    -- KEY tells us which key column has been allocated like PRIMARY/FOREIGN
    -- Default value describes itself
 
CREATE TABLE course(
	id			INT NOT NULL, -- Maintaing column is not left empty
    course_name	VARCHAR(50),
    batch		VARCHAR(30)
    );

DESC course;

INSERT INTO course
	VALUES(101,'DS','Weekdays');
    
INSERT INTO course(course_name,batch)
	VALUES('DM','Weekdays'); -- Error because NULL value not allowed in ID
    
INSERT INTO course(id,course_name)
	VALUES(102,'DM');

SELECT * FROM course;
# ------------------------------------------------------------------------------------

-- Practice 02

USE DSP;

SHOW TABLES;

SELECT * FROM student;
SELECT * FROM course;
DESC student;
DESC course;

INSERT INTO course(id,batch,course_name)
	VALUES(103,'Weekend','DS');
INSERT INTO student
	VALUES
			(108,'Palak','F',23),
            (109,'Vatsal','M',23);
 
# ------------------------------------------------------------------------------------

-- Day 03

USE DSP;

-- Primary Key
	-- Doesn't take null values, all values to be unique
CREATE TABLE employee(
	emp_id		INT PRIMARY KEY,
    emp_name	VARCHAR(50),
    gender		VARCHAR(2),
    designation	VARCHAR(20)
    );

DESC employee;

INSERT INTO employee(emp_name,gender) 
	VALUES('Aarti','F'); -- Error because EmpID(Primary Key) Cannot be Null
    
INSERT INTO employee(emp_id,emp_name,gender)
	VALUES(101,'Aarti','F');
INSERT INTO employee
	VALUES
			(102,'Man','M','DA'),
            (103,'Aadi','M','BA'),
            (104,'Roh','M','DA');

SELECT * FROM employee;

INSERT INTO employee
	VALUES (102,'Man','M','DA'); -- Duplicates in Emp_ID(Primary Key) Not allowed

-- NOT NULL allows duplicates
INSERT INTO course(id)
	VALUES(101); -- Allows Repetition
SELECT * from course; 

CREATE TABLE fees(
	course_id	INT PRIMARY KEY AUTO_INCREMENT,
    fees		INT
    );

INSERT INTO fees
	VALUES (101,25000);
    
INSERT INTO fees(fees)
	VALUES(50000);

SELECT * FROM fees;

CREATE TABLE admin(
	admin_id		int,
    name			VARCHAR(50),
    product			VARCHAR(20),
    product_color	VARCHAR(20) DEFAULT 'red'
    );
    
INSERT INTO admin
	VALUES (101,'Rahul','laptop','golden');
    
INSERT INTO admin(admin_id,name,product)
	VALUES(102,'Tej','mobile');
    
SELECT * FROM admin;

# ------------------------------------------------------------------------------------


