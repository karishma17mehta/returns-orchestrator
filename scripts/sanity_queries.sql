-- Return rate overall
SELECT
  COUNT(r.return_id)::float / NULLIF(COUNT(o.order_id),0) AS return_rate
FROM orders o
LEFT JOIN returns r ON o.order_id = r.order_id;

-- Return rate by category
SELECT
  p.category,
  COUNT(DISTINCT r.order_id) AS returned_orders,
  COUNT(DISTINCT oi.order_id) AS total_orders_with_items,
  COUNT(DISTINCT r.order_id)::float / NULLIF(COUNT(DISTINCT oi.order_id),0) AS return_rate
FROM order_items oi
JOIN products p ON oi.product_id = p.product_id
LEFT JOIN returns r ON oi.order_id = r.order_id
GROUP BY 1
HAVING COUNT(DISTINCT oi.order_id) >= 200
ORDER BY return_rate DESC
LIMIT 15;


-- Late returns share
SELECT
  AVG(CASE WHEN is_late THEN 1 ELSE 0 END) AS late_return_share
FROM returns;