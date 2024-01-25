/*

-----------------------------------------------------------------------------------------------------------------------------------
													    Guidelines
-----------------------------------------------------------------------------------------------------------------------------------

The provided document is a guide for the project. Follow the instructions and take the necessary steps to finish
the project in the SQL file			

-----------------------------------------------------------------------------------------------------------------------------------
                                                         Queries
                                               
-----------------------------------------------------------------------------------------------------------------------------------*/
use vehdb;
  
/*-- QUESTIONS RELATED TO CUSTOMERS
     [Q1] What is the distribution of customers across states?
     Hint: For each state, count the number of customers.*/
     
/*-- SOLUTION:

 - We select the state column from the customer_t table.
 - We use the COUNT(*) function to count the number of customers in each state.
 - We group the results by the state column using GROUP BY.
 - We order the results by customer_count in descending order to show the states with the most customers at the top.

* This query will provide the distribution of customers across states, listing each state along with the number of customers in that state, 
  sorted by the number of customers in descending order. 
*/

SELECT
    state,
    COUNT(*) AS customer_count
FROM
    customer_t
GROUP BY
    state
ORDER BY
    customer_count DESC;

-- ---------------------------------------------------------------------------------------------------------------------------------

/* [Q2] What is the average rating in each quarter?
-- Very Bad is 1, Bad is 2, Okay is 3, Good is 4, Very Good is 5.

Hint: Use a common table expression and in that CTE, assign numbers to the different customer ratings. 
      Now average the feedback for each quarter. */

/*-- SOLUTION:

 To calculate the average rating in each quarter, we can first create a Common Table Expression (CTE) to assign numbers 
 to different customer ratings and then calculate the average feedback for each quarter. Since the ratings and their 
 corresponding values (Very Bad: 1, Bad: 2, Okay: 3, Good: 4, Very Good: 5) are provided, we can use these values in the CTE. 
 
 - The "RatingValues" CTE assigns numerical values to different customer ratings using a CASE statement.
 - We then join the order_t table with the "RatingValues" CTE on the customer_feedback column.
 - We calculate the average rating using the AVG function and group the results by quarter_number.
 - Finally, we order the results by quarter_number.

 * This query will provide the average rating for each quarter based on the numerical values assigned to the different customer feedback ratings.
*/

WITH RatingValues AS (
    SELECT
        customer_feedback,
        CASE
            WHEN customer_feedback = 'Very Bad' THEN 1
            WHEN customer_feedback = 'Bad' THEN 2
            WHEN customer_feedback = 'Okay' THEN 3
            WHEN customer_feedback = 'Good' THEN 4
            WHEN customer_feedback = 'Very Good' THEN 5
            ELSE NULL
        END AS rating_value
    FROM
        order_t
)
SELECT
    quarter_number,
    AVG(rating_value) AS average_rating
FROM
    order_t o
JOIN
    RatingValues r ON o.customer_feedback = r.customer_feedback
GROUP BY
    quarter_number
ORDER BY
    quarter_number;

-- ---------------------------------------------------------------------------------------------------------------------------------

/* [Q3] Are customers getting more dissatisfied over time?

Hint: Need the percentage of different types of customer feedback in each quarter. Use a common table expression and
	  determine the number of customer feedback in each category as well as the total number of customer feedback in each quarter.
	  Now use that common table expression to find out the percentage of different types of customer feedback in each quarter.
      Eg: (total number of very good feedback/total customer feedback)* 100 gives you the percentage of very good feedback.
*/

/*-- SOLUTION:

 To determine whether customers are getting more dissatisfied over time, we can calculate 
 the percentage of different types of customer feedback in each quarter and then analyze the trends. 

 - The FeedbackCounts CTE calculates the count of each type of customer feedback in each quarter.
 - The QuarterTotal CTE calculates the total count of feedback for each quarter.
 - The final SELECT statement joins the two CTEs and calculates the percentage for each type of 
   customer feedback in each quarter using the formula you provided.

  By analyzing the results from this query, we can assess whether customers are getting more dissatisfied 
  over time by looking at the percentage of "Very Bad" and "Bad" feedback and how it changes across quarters.
  
*/
 

WITH FeedbackCounts AS (
    SELECT
        quarter_number,
        customer_feedback,
        COUNT(*) AS feedback_count
    FROM
        order_t
    GROUP BY
        quarter_number,
        customer_feedback
),
QuarterTotal AS (
    SELECT
        quarter_number,
        SUM(feedback_count) AS total_count
    FROM
        FeedbackCounts
    GROUP BY
        quarter_number
)
SELECT
    Q1.quarter_number,
    Q1.customer_feedback,
    Q1.feedback_count,
    Q2.total_count,
    (Q1.feedback_count * 100.0 / Q2.total_count) AS feedback_percentage
FROM
    FeedbackCounts Q1
JOIN
    QuarterTotal Q2 ON Q1.quarter_number = Q2.quarter_number
ORDER BY
    Q1.quarter_number, Q1.customer_feedback;

      
-- ---------------------------------------------------------------------------------------------------------------------------------

/*[Q4] Which are the top 5 vehicle makers preferred by the customer.

Hint: For each vehicle make what is the count of the customers.*/

/*-- SOLUTION:

 - We select the vehicle_maker column from the order_t table.
 - We use the COUNT(DISTINCT customer_id) function to count the number of distinct customers for each vehicle make. 
   This ensures that each customer is counted only once for a specific vehicle make.
 - We group the results by the vehicle_maker column using GROUP BY.
 - We order the results by customer_count in descending order to identify the top 5 vehicle makers preferred by customers.

 The query will give us the top 5 vehicle makers preferred by customers, along with the count of customers
 who have purchased vehicles from each of those makers.
  
*/

SELECT
    vehicle_maker,
    COUNT(DISTINCT customer_id) AS customer_count
FROM
    product_t p
    LEFT JOIN order_t o ON p.product_id = o.product_id
GROUP BY
    vehicle_maker
ORDER BY
    customer_count DESC
LIMIT 5;


-- ---------------------------------------------------------------------------------------------------------------------------------

/*[Q5] What is the most preferred vehicle make in each state?

Hint: Use the window function RANK() to rank based on the count of customers for each state and vehicle maker. 
After ranking, take the vehicle maker whose rank is 1.*/

/*-- SOLUTION:

 To find the most preferred vehicle make in each state, we can use the RANK() window function to rank based on 
 the count of customers for each state and vehicle maker, and then select the vehicle maker with a rank of 1.

  - We create the StateVehicleCounts CTE, which calculates the count of distinct customers for each state and vehicle maker, 
    and use the RANK() window function to rank the results within each state in descending order.
  - We then select the rows where the rank is equal to 1, which corresponds to the most preferred vehicle maker in each state.

  The query will provide us with the most preferred vehicle make in each state based on the count of customers who have purchased vehicles from each maker.  
  
*/


WITH StateVehicleCounts AS (
    SELECT
        c.state AS state,
        p.vehicle_maker AS vehicle_maker,
        COUNT(DISTINCT c.customer_id) AS customer_count,
        RANK() OVER (PARTITION BY c.state ORDER BY COUNT(DISTINCT c.customer_id) DESC) AS customer_rank
    FROM
        order_t o
        JOIN customer_t c ON o.customer_id = c.customer_id
        JOIN product_t p ON o.product_id = p.product_id
    GROUP BY
        c.state,
        p.vehicle_maker
)
SELECT
    state,
    vehicle_maker
FROM
    StateVehicleCounts
WHERE
    customer_rank = 1;


-- ---------------------------------------------------------------------------------------------------------------------------------

/*QUESTIONS RELATED TO REVENUE and ORDERS 

-- [Q6] What is the trend of number of orders by quarters?

Hint: Count the number of orders for each quarter.*/

/*-- SOLUTION:

To determine the trend of the number of orders by quarters, we can count the number of orders for each quarter. 

  - We select the quarter_number column from the order_t table.
  - We use the COUNT(*) function to count the number of orders in each quarter.
  - We group the results by the quarter_number column using GROUP BY.
  - We order the results by quarter_number to show the trend of the number of orders by quarters.

 This query will provide us with the trend of the number of orders in each quarter, allowing us to see how the order volume changes over time.  

*/

SELECT
    quarter_number,
    COUNT(*) AS order_count
FROM
    order_t
GROUP BY
    quarter_number
ORDER BY
    quarter_number;


-- ---------------------------------------------------------------------------------------------------------------------------------

/* [Q7] What is the quarter over quarter % change in revenue? 

Hint: Quarter over Quarter percentage change in revenue means what is the change in revenue from the subsequent quarter to the previous quarter in percentage.
      To calculate you need to use the common table expression to find out the sum of revenue for each quarter.
      Then use that CTE along with the LAG function to calculate the QoQ percentage change in revenue.
*/

/*-- SOLUTION:

 To calculate the quarter-over-quarter percentage change in revenue, we can use a Common Table Expression (CTE) 
 to find the sum of revenue for each quarter and then use the LAG() function to calculate the percentage change from one quarter to the next. 

 - The QuarterlyRevenue CTE calculates the sum of revenue for each quarter by summing the vehicle_price column in the order_t table.
 - We use the LAG() function to retrieve the revenue for the previous quarter.
 - We calculate the quarter-over-quarter percentage change in revenue by comparing the revenue of the current quarter 
   with the revenue of the previous quarter and expressing it as a percentage change.

 This query will provide us with the quarter-over-quarter percentage change in revenue, showing how revenue changes from one quarter to the next.

*/
      
WITH QuarterlyRevenue AS (
    SELECT
        quarter_number,
        SUM(vehicle_price) AS revenue
    FROM
        order_t
    GROUP BY
        quarter_number
    ORDER BY
        quarter_number
)
SELECT
    qr1.quarter_number,
    qr1.revenue AS current_quarter_revenue,
    LAG(qr1.revenue) OVER (ORDER BY qr1.quarter_number) AS previous_quarter_revenue,
    ((qr1.revenue - LAG(qr1.revenue) OVER (ORDER BY qr1.quarter_number)) / LAG(qr1.revenue) OVER (ORDER BY qr1.quarter_number)) * 100 AS qoq_percentage_change
FROM
    QuarterlyRevenue qr1;
      

-- ---------------------------------------------------------------------------------------------------------------------------------

/* [Q8] What is the trend of revenue and orders by quarters?

Hint: Find out the sum of revenue and count the number of orders for each quarter.*/

/*-- SOLUTION:

 To find the trend of revenue and the number of orders by quarters, we can calculate the sum of revenue 
 and count the number of orders for each quarter.
 
 - The QuarterlySummary CTE calculates the sum of revenue (using SUM(vehicle_price)) and counts the number of orders (using COUNT(*)) for each quarter.
 - The results are grouped by quarter_number and ordered by quarter_number.
 - The final SELECT statement retrieves the quarter number, total revenue, and total number of orders from the QuarterlySummary CTE.

This query will provide us with the trend of revenue and the number of orders by quarters, allowing us to see how revenue and order volume change over time.

*/

WITH QuarterlySummary AS (
    SELECT
        quarter_number,
        SUM(vehicle_price) AS total_revenue,
        COUNT(*) AS total_orders
    FROM
        order_t
    GROUP BY
        quarter_number
    ORDER BY
        quarter_number
)
SELECT
    qs.quarter_number,
    qs.total_revenue,
    qs.total_orders
FROM
    QuarterlySummary qs;


-- ---------------------------------------------------------------------------------------------------------------------------------

/* QUESTIONS RELATED TO SHIPPING 
    [Q9] What is the average discount offered for different types of credit cards?

Hint: Find out the average of discount for each credit card type.*/

/*-- SOLUTION:

To find the average discount offered for different types of credit cards, we can calculate the average discount for each credit card type.
 
 - We select the credit_card_type column from the order_t table.
 - We use the AVG(discount) function to calculate the average discount for each credit card type.
 - We group the results by the credit_card_type using GROUP BY.

This query will provide us with the average discount offered for different types of credit cards, 
helping you understand the average discount associated with each credit card type.

*/

SELECT
    c.credit_card_type AS credit_card_type,
    AVG(o.discount) AS average_discount
FROM
    order_t o
    JOIN customer_t c ON o.customer_id = c.customer_id
GROUP BY
    credit_card_type;


-- ---------------------------------------------------------------------------------------------------------------------------------

/* [Q10] What is the average time taken to ship the placed orders for each quarters?
	Hint: Use the dateiff function to find the difference between the ship date and the order date.
*/

/*-- SOLUTION:

To find the average time taken to ship placed orders for each quarter, you can use the DATEDIFF function 
to calculate the difference between the ship date and the order date and then calculate the average time for each quarter.

- We select the quarter_number column from the order_t table.
- We use the DATEDIFF(day, order_date, ship_date) function to calculate the difference in days between the order date and ship date for each order.
- We use AVG to calculate the average shipping time in days for each quarter.
- We group the results by the quarter_number using GROUP BY.

This query will provide us with the average time taken to ship placed orders for each quarter, helping us analyze the shipping efficiency over time.

*/

SELECT
    quarter_number,
    AVG(DATEDIFF(ship_date, order_date)) AS average_shipping_time
FROM
    order_t
GROUP BY
    quarter_number
ORDER BY
	quarter_number;

-- --------------------------------------------------------Done----------------------------------------------------------------------
-- ----------------------------------------------------------------------------------------------------------------------------------

