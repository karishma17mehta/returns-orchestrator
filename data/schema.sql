DROP TABLE IF EXISTS returns;
DROP TABLE IF EXISTS order_items;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS products;

CREATE TABLE products (
  product_id TEXT PRIMARY KEY,
  category TEXT,
  price NUMERIC
);

CREATE TABLE orders (
  order_id TEXT PRIMARY KEY,
  customer_id TEXT,
  order_date DATE,
  delivered_date DATE,
  status TEXT,
  total_amount NUMERIC
);

CREATE TABLE order_items (
  order_id TEXT,
  order_item_id INT,
  product_id TEXT,
  quantity INT,
  item_price NUMERIC,
  PRIMARY KEY (order_id, order_item_id)
);

CREATE TABLE returns (
  return_id TEXT PRIMARY KEY,
  order_id TEXT,
  return_date DATE,
  reason TEXT,
  refund_amount NUMERIC,
  is_late BOOLEAN
);

ALTER TABLE order_items
  ADD CONSTRAINT fk_order_items_orders FOREIGN KEY (order_id) REFERENCES orders(order_id);

ALTER TABLE order_items
  ADD CONSTRAINT fk_order_items_products FOREIGN KEY (product_id) REFERENCES products(product_id);

ALTER TABLE returns
  ADD CONSTRAINT fk_returns_orders FOREIGN KEY (order_id) REFERENCES orders(order_id);


CREATE INDEX idx_orders_order_date ON orders(order_date);
CREATE INDEX idx_returns_return_date ON returns(return_date);
CREATE INDEX idx_products_category ON products(category);
CREATE INDEX idx_order_items_order_id ON order_items(order_id);
CREATE INDEX idx_order_items_product_id ON order_items(product_id);
