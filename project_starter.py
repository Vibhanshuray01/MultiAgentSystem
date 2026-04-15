import pandas as pd
import numpy as np
import os
import time
import dotenv
import ast
import json
import requests
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Union
from sqlalchemy import create_engine, Engine

# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")

# List containing the different kinds of papers 
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper",                         "category": "paper",        "unit_price": 0.05},
    {"item_name": "Letter-sized paper",              "category": "paper",        "unit_price": 0.06},
    {"item_name": "Cardstock",                        "category": "paper",        "unit_price": 0.15},
    {"item_name": "Colored paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Glossy paper",                     "category": "paper",        "unit_price": 0.20},
    {"item_name": "Matte paper",                      "category": "paper",        "unit_price": 0.18},
    {"item_name": "Recycled paper",                   "category": "paper",        "unit_price": 0.08},
    {"item_name": "Eco-friendly paper",               "category": "paper",        "unit_price": 0.12},
    {"item_name": "Poster paper",                     "category": "paper",        "unit_price": 0.25},
    {"item_name": "Banner paper",                     "category": "paper",        "unit_price": 0.30},
    {"item_name": "Kraft paper",                      "category": "paper",        "unit_price": 0.10},
    {"item_name": "Construction paper",               "category": "paper",        "unit_price": 0.07},
    {"item_name": "Wrapping paper",                   "category": "paper",        "unit_price": 0.15},
    {"item_name": "Glitter paper",                    "category": "paper",        "unit_price": 0.22},
    {"item_name": "Decorative paper",                 "category": "paper",        "unit_price": 0.18},
    {"item_name": "Letterhead paper",                 "category": "paper",        "unit_price": 0.12},
    {"item_name": "Legal-size paper",                 "category": "paper",        "unit_price": 0.08},
    {"item_name": "Crepe paper",                      "category": "paper",        "unit_price": 0.05},
    {"item_name": "Photo paper",                      "category": "paper",        "unit_price": 0.25},
    {"item_name": "Uncoated paper",                   "category": "paper",        "unit_price": 0.06},
    {"item_name": "Butcher paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Heavyweight paper",                "category": "paper",        "unit_price": 0.20},
    {"item_name": "Standard copy paper",              "category": "paper",        "unit_price": 0.04},
    {"item_name": "Bright-colored paper",             "category": "paper",        "unit_price": 0.12},
    {"item_name": "Patterned paper",                  "category": "paper",        "unit_price": 0.15},

    # Product Types (priced per unit)
    {"item_name": "Paper plates",                     "category": "product",      "unit_price": 0.10},  # per plate
    {"item_name": "Paper cups",                       "category": "product",      "unit_price": 0.08},  # per cup
    {"item_name": "Paper napkins",                    "category": "product",      "unit_price": 0.02},  # per napkin
    {"item_name": "Disposable cups",                  "category": "product",      "unit_price": 0.10},  # per cup
    {"item_name": "Table covers",                     "category": "product",      "unit_price": 1.50},  # per cover
    {"item_name": "Envelopes",                        "category": "product",      "unit_price": 0.05},  # per envelope
    {"item_name": "Sticky notes",                     "category": "product",      "unit_price": 0.03},  # per sheet
    {"item_name": "Notepads",                         "category": "product",      "unit_price": 2.00},  # per pad
    {"item_name": "Invitation cards",                 "category": "product",      "unit_price": 0.50},  # per card
    {"item_name": "Flyers",                           "category": "product",      "unit_price": 0.15},  # per flyer
    {"item_name": "Party streamers",                  "category": "product",      "unit_price": 0.05},  # per roll
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},  # per roll
    {"item_name": "Paper party bags",                 "category": "product",      "unit_price": 0.25},  # per bag
    {"item_name": "Name tags with lanyards",          "category": "product",      "unit_price": 0.75},  # per tag
    {"item_name": "Presentation folders",             "category": "product",      "unit_price": 0.50},  # per folder

    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    # Specialty papers
    {"item_name": "100 lb cover stock",               "category": "specialty",    "unit_price": 0.50},
    {"item_name": "80 lb text paper",                 "category": "specialty",    "unit_price": 0.40},
    {"item_name": "250 gsm cardstock",                "category": "specialty",    "unit_price": 0.30},
    {"item_name": "220 gsm poster paper",             "category": "specialty",    "unit_price": 0.35},
]

# Given below are some utility functions you can use to implement your multi-agent system

def generate_sample_inventory(paper_supplies: list, coverage: float = 0.4, seed: int = 137) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
                               keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                      - item_name
                      - category
                      - unit_price
                      - current_stock
                      - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)),
        size=num_items,
        replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": np.random.randint(200, 800),  # Realistic stock range
            "min_stock_level": np.random.randint(50, 150)  # Reasonable threshold for reordering
        })

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)

def init_database(db_engine: Engine, seed: int = 137) -> Engine:    
    """
    Set up the Munder Difflin database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                              Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame({
            "id": [],
            "item_name": [],
            "transaction_type": [],  # 'stock_orders' or 'sales'
            "units": [],             # Quantity involved
            "price": [],             # Total price for the transaction
            "transaction_date": [],  # ISO-formatted date
        })
        transactions_schema.to_sql("transactions", db_engine, if_exists="replace", index=False)

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", ""))
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", ""))
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", ""))

        # Retain only relevant columns
        quotes_df = quotes_df[[
            "request_id",
            "total_amount",
            "quote_explanation",
            "order_date",
            "job_type",
            "order_size",
            "event_type"
        ]]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append({
            "item_name": None,
            "transaction_type": "sales",
            "units": None,
            "price": 50000.0,
            "transaction_date": initial_date,
        })

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append({
                "item_name": item["item_name"],
                "transaction_type": "stock_orders",
                "units": item["current_stock"],
                "price": item["current_stock"] * item["unit_price"],
                "transaction_date": initial_date,
            })

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql("transactions", db_engine, if_exists="append", index=False)

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame([{
            "item_name": item_name,
            "transaction_type": transaction_type,
            "units": quantity,
            "price": price,
            "transaction_date": date_str,
        }])

        # Insert the record into the database
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)

        # Fetch and return the ID of the inserted row
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise

def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing 
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))

def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and 
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE LOWER(item_name) = LOWER(:item_name)
        AND transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )

def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    print(f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'")

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.")
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")

def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0


def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": item["unit_price"],
            "value": item_value,
        })

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }


def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result]

########################
########################
########################
# YOUR MULTI AGENT STARTS HERE
########################
########################
########################


# Set up and load your env parameters and instantiate your model.

dotenv.load_dotenv()

VOCAREUM_API_KEY  = os.getenv("UDACITY_OPENAI_API_KEY",
                               "voc-630438504158766489910469dd0321f111b9.75557243")
VOCAREUM_BASE_URL = "https://openai.vocareum.com/v1"
MODEL_NAME        = "gpt-4o-mini"


def call_llm(messages: List[Dict], tools: List[Dict] = None,
             model: str = MODEL_NAME, temperature: float = 0) -> Dict:
    """
    Call the Vocareum OpenAI-compatible API and return the response JSON.

    Enables OpenAI function-calling when `tools` is supplied.
    """
    body = {"model": model, "messages": messages, "temperature": temperature}
    if tools:
        body["tools"]       = tools
        body["tool_choice"] = "auto"
    resp = requests.post(
        f"{VOCAREUM_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {VOCAREUM_API_KEY}",
                 "Content-Type": "application/json"},
        json=body,
        timeout=90,
    )
    resp.raise_for_status()
    return resp.json()


class Agent:
    """
    A general-purpose tool-calling agent.

    Runs a ReAct-style loop:
      1. Call the LLM with the current message history and tool schemas.
      2. If the LLM emits tool_calls, execute each one and append the results.
      3. Repeat until the LLM returns plain text with no tool_calls.

    Attributes:
        name           – Human-readable identifier (used in log output).
        system_prompt  – Injected as the system message on every run.
        tools          – OpenAI function-calling schemas for this agent.
        tool_registry  – Maps tool name → Python callable.
        max_iterations – Safety limit for the tool loop (default 10).
    """

    def __init__(self, name: str, system_prompt: str,
                 tools: List[Dict], tool_registry: Dict,
                 max_iterations: int = 10):
        self.name           = name
        self.system_prompt  = system_prompt
        self.tools          = tools
        self.tool_registry  = tool_registry
        self.max_iterations = max_iterations

    def run(self, user_message: str) -> str:
        """Run the tool-calling loop and return the final plain-text answer."""
        print(f"\n  [{self.name}] TASK: {user_message[:110]}...")
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user",   "content": user_message}]

        for iteration in range(self.max_iterations):
            resp = call_llm(messages, tools=self.tools or None)
            msg  = resp["choices"][0]["message"]
            messages.append(msg)

            tool_calls = msg.get("tool_calls")
            if not tool_calls:
                print(f"  [{self.name}] DONE in {iteration + 1} iteration(s).")
                return msg.get("content", "No response generated.")

            for call in tool_calls:
                fn_name = call["function"]["name"]
                try:
                    fn_args = json.loads(call["function"]["arguments"])
                except json.JSONDecodeError:
                    fn_args = {}
                print(f"  [{self.name}] → {fn_name}({fn_args})")
                if fn_name in self.tool_registry:
                    try:
                        result = self.tool_registry[fn_name](**fn_args)
                    except Exception as exc:
                        result = f"TOOL ERROR ({fn_name}): {exc}"
                else:
                    result = f"UNKNOWN TOOL: '{fn_name}'"
                messages.append({"role": "tool", "tool_call_id": call["id"],
                                  "content": str(result)})

        return "Maximum iterations reached. Please simplify the request."


"""Set up tools for your agents to use, these should be methods that combine the database functions above
 and apply criteria to them to ensure that the flow of the system is correct."""


# Tools for inventory agent

def tool_get_catalog(date: str) -> str:
    """List all items currently in stock with their quantities. Uses get_all_inventory()."""
    inv = get_all_inventory(date)
    if not inv:
        return "No items are currently available in inventory."
    lines = [f"  - {name}: {qty:,} units" for name, qty in sorted(inv.items())]
    return f"AVAILABLE CATALOGUE (as of {date}):\n" + "\n".join(lines)


def tool_check_item_stock(item_name: str, date: str) -> str:
    """Check stock for one specific item by exact catalogue name. Uses get_stock_level()."""
    df    = get_stock_level(item_name, date)
    stock = int(df["current_stock"].iloc[0])
    label = "IN STOCK" if stock > 0 else "OUT OF STOCK"
    return f"'{item_name}': {stock:,} units as of {date} [{label}]"


def tool_reorder_item(item_name: str, quantity: int, order_date: str) -> str:
    """
    Place a supplier reorder for an item that is low in stock.
    Uses get_cash_balance, get_supplier_delivery_date, and create_transaction.
    """
    price_df = pd.read_sql(
        # "SELECT unit_price FROM inventory WHERE item_name = :n",
        "SELECT unit_price FROM inventory WHERE LOWER(item_name) = LOWER(:n)",
        db_engine, params={"n": item_name}
    )
    if price_df.empty:
        return f"ERROR: '{item_name}' not found in catalogue. Cannot reorder."
    unit_price = float(price_df["unit_price"].iloc[0])
    total_cost = unit_price * quantity
    cash       = get_cash_balance(order_date)
    if cash < total_cost:
        return (f"INSUFFICIENT FUNDS: reorder of {quantity:,}×'{item_name}' "
                f"costs ${total_cost:,.2f}; cash balance is ${cash:,.2f}.")
    delivery = get_supplier_delivery_date(order_date, quantity)
    txn_id   = create_transaction(item_name, "stock_orders", quantity, total_cost, delivery)
    return (f"REORDER PLACED: {quantity:,}×'{item_name}' @${unit_price:.2f}/unit. "
            f"Total ${total_cost:,.2f}. Delivery: {delivery}. Transaction ID: {txn_id}.")


INVENTORY_TOOLS_SCHEMA = [
    {"type": "function", "function": {
        "name": "get_catalog",
        "description": (
            "List ALL paper products currently in stock with quantities. "
            "Always call this first to see available items and their EXACT catalogue names."),
        "parameters": {"type": "object",
                       "properties": {"date": {"type": "string",
                                               "description": "YYYY-MM-DD"}},
                       "required": ["date"]}}},
    {"type": "function", "function": {
        "name": "check_item_stock",
        "description": "Check current stock for one specific catalogue item by exact name.",
        "parameters": {"type": "object",
                       "properties": {
                           "item_name": {"type": "string",
                                         "description": "Exact catalogue name (case-sensitive)."},
                           "date":      {"type": "string", "description": "YYYY-MM-DD"}},
                       "required": ["item_name", "date"]}}},
    {"type": "function", "function": {
        "name": "reorder_item",
        "description": (
            "Place a supplier reorder for a low-stock item. "
            "Records a stock_orders transaction and returns the delivery date."),
        "parameters": {"type": "object",
                       "properties": {
                           "item_name":  {"type": "string"},
                           "quantity":   {"type": "integer"},
                           "order_date": {"type": "string", "description": "YYYY-MM-DD"}},
                       "required": ["item_name", "quantity", "order_date"]}}},
]


# Tools for quoting agent

def tool_find_similar_quotes(keywords: str) -> str:
    """Search historical quotes for pricing context. Uses search_quote_history()."""
    terms = [k.strip() for k in keywords.split(",") if k.strip()]
    if not terms:
        return "No keywords provided."
    results = search_quote_history(terms, limit=5)
    if not results:
        return "No similar historical quotes found."
    lines = ["HISTORICAL QUOTES (for context):"]
    for i, q in enumerate(results, 1):
        expl = str(q.get("quote_explanation", ""))[:200]
        lines.append(f"\n[{i}] Total: ${q.get('total_amount','?')} | "
                     f"Job: {q.get('job_type','?')} | "
                     f"Size: {q.get('order_size','?')} | "
                     f"Event: {q.get('event_type','?')}")
        lines.append(f"     {expl}...")
    return "\n".join(lines)


def tool_build_quote(items_quantities_json: str, request_date: str) -> str:
    """
    Generate a price quote with bulk discounts.

    Discount tiers (total units across all items):
      ≤500 units  →  5% | 501–2000 → 10% | >2000 → 15%

    Uses get_stock_level() for availability checks and the inventory table for unit prices.
    Returns a machine-readable QUOTE_DATA_JSON block for the sales agent.
    """
    try:
        items: Dict[str, int] = json.loads(items_quantities_json)
    except (json.JSONDecodeError, ValueError) as exc:
        return f"ERROR: invalid JSON — {exc}. Example: '{{\"A4 paper\": 500}}'"
    if not items:
        return "ERROR: no items provided."

    line_items, unavailable = [], []
    subtotal = total_units = 0.0

    for item_name, raw_qty in items.items():
        qty   = int(raw_qty)
        stock = int(get_stock_level(item_name, request_date)["current_stock"].iloc[0])
        if stock < qty:
            unavailable.append(f"{item_name}: need {qty:,}, only {stock:,} in stock")
            continue
        price_df = pd.read_sql(
            # "SELECT unit_price FROM inventory WHERE item_name = :n",
            "SELECT unit_price FROM inventory WHERE LOWER(item_name) = LOWER(:n)",

            db_engine, params={"n": item_name}
        )
        if price_df.empty:
            unavailable.append(f"{item_name}: not found in catalogue")
            continue
        up         = float(price_df["unit_price"].iloc[0])
        lt         = up * qty
        subtotal  += lt
        total_units += qty
        line_items.append({"item": item_name, "quantity": qty,
                           "unit_price": up, "line_total": lt})

    if not line_items:
        msg = "QUOTE: no items could be quoted.\n"
        if unavailable:
            msg += "Issues:\n" + "\n".join(f"  - {u}" for u in unavailable)
        return msg

    if total_units <= 500:
        disc, label = 0.05, "5% (≤500 units)"
    elif total_units <= 2000:
        disc, label = 0.10, "10% (501–2,000 units)"
    else:
        disc, label = 0.15, "15% (>2,000 units)"

    disc_amt = subtotal * disc
    total    = round(subtotal - disc_amt, 2)

    lines = [f"=== PRICE QUOTE ({request_date}) ==="]
    for li in line_items:
        lines.append(f"  {li['item']}: {li['quantity']:,} × ${li['unit_price']:.2f}"
                     f" = ${li['line_total']:.2f}")
    lines += [f"Subtotal:               ${subtotal:.2f}",
              f"Bulk discount ({label}): -${disc_amt:.2f}",
              f"TOTAL DUE:              ${total:.2f}",
              f"Units ordered:          {int(total_units):,}"]
    if unavailable:
        lines.append("\nExcluded (out of stock / not in catalogue):")
        lines += [f"  - {u}" for u in unavailable]
    lines.append(f"\nQUOTE_DATA_JSON:{json.dumps({'items': line_items, 'total': total})}")
    return "\n".join(lines)


QUOTE_TOOLS_SCHEMA = [
    {"type": "function", "function": {
        "name": "find_similar_quotes",
        "description": "Search historical quotes for pricing context.",
        "parameters": {"type": "object",
                       "properties": {"keywords": {"type": "string",
                                                    "description": "Comma-separated search terms."}},
                       "required": ["keywords"]}}},
    {"type": "function", "function": {
        "name": "build_quote",
        "description": (
            "Generate a price quote for a set of items. "
            "Bulk discounts: 5% for ≤500 units, 10% for 501–2000, 15% for >2000. "
            "Returns a QUOTE_DATA_JSON block for the sales agent to use."),
        "parameters": {"type": "object",
                       "properties": {
                           "items_quantities_json": {
                               "type": "string",
                               "description": (
                                   "JSON object: exact catalogue names → integer quantities. "
                                   "E.g. '{\"Cardstock\": 200, \"Glossy paper\": 500}'")},
                           "request_date": {"type": "string",
                                            "description": "YYYY-MM-DD"}},
                       "required": ["items_quantities_json", "request_date"]}}},
]


# Tools for ordering agent

def tool_process_sale(item_name: str, quantity: int,
                       total_price: float, sale_date: str) -> str:
    """
    Record a completed sale for one line item.
    Uses get_stock_level() to verify availability and create_transaction() to record.
    """
    stock = int(get_stock_level(item_name, sale_date)["current_stock"].iloc[0])
    if stock < quantity:
        return (f"SALE FAILED: '{item_name}' — need {quantity:,}, "
                f"only {stock:,} available.")
    txn_id = create_transaction(item_name, "sales", quantity, total_price, sale_date)
    return (f"SALE RECORDED ✓  {quantity:,}×'{item_name}' for "
            f"${total_price:.2f} on {sale_date}. Transaction ID: {txn_id}.")


def tool_get_cash_balance_info(date: str) -> str:
    """Return the current cash balance. Uses get_cash_balance()."""
    return f"Cash balance as of {date}: ${get_cash_balance(date):,.2f}"


def tool_get_delivery_estimate(order_date: str, quantity: int) -> str:
    """Estimate delivery date for an order. Uses get_supplier_delivery_date()."""
    d = get_supplier_delivery_date(order_date, quantity)
    return f"Estimated delivery for {quantity:,} units ordered on {order_date}: {d}."


def tool_get_financial_summary(date: str) -> str:
    """Return a financial summary. Uses generate_financial_report()."""
    r = generate_financial_report(date)
    lines = [f"=== FINANCIAL SUMMARY ({date}) ===",
             f"Cash:      ${r['cash_balance']:,.2f}",
             f"Inventory: ${r['inventory_value']:,.2f}",
             f"Assets:    ${r['total_assets']:,.2f}"]
    for p in r.get("top_selling_products", []):
        if p.get("item_name"):
            lines.append(f"  Top — {p['item_name']}: "
                         f"{p.get('total_units',0):,} units / ${p.get('total_revenue',0):.2f}")
    return "\n".join(lines)


SALES_TOOLS_SCHEMA = [
    {"type": "function", "function": {
        "name": "process_sale",
        "description": (
            "Record a confirmed sale for ONE catalogue item. "
            "Call once per line item. Verifies stock before recording."),
        "parameters": {"type": "object",
                       "properties": {
                           "item_name":   {"type": "string"},
                           "quantity":    {"type": "integer"},
                           "total_price": {"type": "number",
                                           "description": "Line total after discount."},
                           "sale_date":   {"type": "string", "description": "YYYY-MM-DD"}},
                       "required": ["item_name", "quantity", "total_price", "sale_date"]}}},
    {"type": "function", "function": {
        "name": "get_cash_balance_info",
        "description": "Check the company cash balance.",
        "parameters": {"type": "object",
                       "properties": {"date": {"type": "string"}},
                       "required": ["date"]}}},
    {"type": "function", "function": {
        "name": "get_delivery_estimate",
        "description": "Estimate delivery date based on order quantity and date.",
        "parameters": {"type": "object",
                       "properties": {
                           "order_date": {"type": "string"},
                           "quantity":   {"type": "integer"}},
                       "required": ["order_date", "quantity"]}}},
    {"type": "function", "function": {
        "name": "get_financial_summary",
        "description": "Full financial summary (cash + inventory value).",
        "parameters": {"type": "object",
                       "properties": {"date": {"type": "string"}},
                       "required": ["date"]}}},
]


# Set up your agents and create an orchestration agent that will manage them.

# ── System prompts ──────────────────────────────────────────────────────

INVENTORY_SYSTEM_PROMPT = """
You are the Inventory Manager for Beaver's Choice Paper Company.

- Call get_catalog first to see all available items and their exact names.
- Use check_item_stock for specific item queries.
- Call reorder_item when post-sale stock will fall below ~150 units.
- Always report exact catalogue names (case-sensitive) in your responses.
- If an item is absent from the catalogue, state clearly it is unavailable.
""".strip()

QUOTE_SYSTEM_PROMPT = """
You are the Quoting Specialist for Beaver's Choice Paper Company.

DISCOUNT TIERS (total units across all items):
  ≤500 units  →  5%  |  501–2,000 → 10%  |  >2,000 → 15%

1. Optionally call find_similar_quotes for historical pricing context.
2. Call build_quote with EXACT catalogue item names and integer quantities.
3. Return the full formatted quote with line items, discount, and total.
4. If some items are unavailable, still quote what IS available.
""".strip()

SALES_SYSTEM_PROMPT = """
You are the Sales Operations Manager for Beaver's Choice Paper Company.

- Call process_sale ONCE PER item using the line_total from the quote.
- Call get_delivery_estimate for the total quantity to provide a delivery date.
- Confirm all transaction IDs and delivery date in your response.
- Do NOT expose internal costs or profit margins.
- If a sale fails, report it and continue processing remaining items.
""".strip()

ORCHESTRATOR_SYSTEM_PROMPT = """
You are the Customer Service Orchestrator for Beaver's Choice Paper Company.
Handle every customer request end-to-end by coordinating three specialist agents.

MANDATORY WORKFLOW:
1. CATALOGUE + STOCK
   Call ask_inventory_agent: "Get full catalogue for [date]. Check stock for: [items]"
   Use the response to map customer descriptions to EXACT catalogue names.

2. QUOTE
   Call ask_quote_agent: "Generate quote for [date]. Items: {exact_name: qty, ...}.
   Customer: [role], Event: [event]."

3. FINALISE SALE (if ≥1 item is available)
   Call ask_sales_agent: "Process sale on [date]. Items: [name: qty at $line_total, ...]"

4. CUSTOMER RESPONSE — compose a professional, friendly reply that includes:
   • Greeting referencing their event / context
   • Items fulfilled: name, quantity, total price
   • Discount tier explanation
   • Expected delivery date
   • Polite explanation for any unfulfilled items

RULES:
- Do NOT reveal internal unit costs or profit margins.
- Do NOT expose raw system errors — rephrase politely.
- Map ALL customer item names to exact catalogue names BEFORE quoting.
- If NOTHING can be fulfilled, still respond politely with an explanation.
- Always pass the request date to all delegated tasks.
""".strip()

# ── Orchestrator tool schemas ───────────────────────────────────────────

ORCHESTRATOR_TOOLS_SCHEMA = [
    {"type": "function", "function": {
        "name": "ask_inventory_agent",
        "description": (
            "Delegate to the Inventory Agent. Use for: "
            "(1) getting the full catalogue (do this FIRST to get exact item names), "
            "(2) checking stock for specific items, "
            "(3) placing supplier reorders."),
        "parameters": {"type": "object",
                       "properties": {"task": {"type": "string",
                                               "description": "Task description with date and items."}},
                       "required": ["task"]}}},
    {"type": "function", "function": {
        "name": "ask_quote_agent",
        "description": (
            "Delegate to the Quote Agent. "
            "Include EXACT catalogue item names, quantities, date, and customer context."),
        "parameters": {"type": "object",
                       "properties": {"task": {"type": "string"}},
                       "required": ["task"]}}},
    {"type": "function", "function": {
        "name": "ask_sales_agent",
        "description": (
            "Delegate to the Sales Agent to finalise a confirmed sale. "
            "Provide exact item names, quantities, line totals, and sale date. "
            "Only call AFTER a valid quote has been obtained."),
        "parameters": {"type": "object",
                       "properties": {"task": {"type": "string"}},
                       "required": ["task"]}}},
]


def create_agent_system() -> Agent:
    """
    Instantiate all four agents and wire them together.

    Workers (InventoryAgent, QuoteAgent, SalesAgent) are wrapped as Python
    closures and registered as tools in the Orchestrator's tool_registry.
    The Orchestrator delegates by calling ask_inventory_agent /
    ask_quote_agent / ask_sales_agent exactly like any other tool.

    Returns:
        Agent: The configured OrchestratorAgent, ready to receive requests.
    """
    inventory_agent = Agent(
        name="InventoryAgent",
        system_prompt=INVENTORY_SYSTEM_PROMPT,
        tools=INVENTORY_TOOLS_SCHEMA,
        tool_registry={
            "get_catalog":      tool_get_catalog,
            "check_item_stock": tool_check_item_stock,
            "reorder_item":     tool_reorder_item,
        },
    )
    quote_agent = Agent(
        name="QuoteAgent",
        system_prompt=QUOTE_SYSTEM_PROMPT,
        tools=QUOTE_TOOLS_SCHEMA,
        tool_registry={
            "find_similar_quotes": tool_find_similar_quotes,
            "build_quote":         tool_build_quote,
        },
    )
    sales_agent = Agent(
        name="SalesAgent",
        system_prompt=SALES_SYSTEM_PROMPT,
        tools=SALES_TOOLS_SCHEMA,
        tool_registry={
            "process_sale":          tool_process_sale,
            "get_cash_balance_info": tool_get_cash_balance_info,
            "get_delivery_estimate": tool_get_delivery_estimate,
            "get_financial_summary": tool_get_financial_summary,
        },
    )

    # Managed-agent shims: expose each worker as a plain callable tool
    def ask_inventory_agent(task: str) -> str:
        return inventory_agent.run(task)

    def ask_quote_agent(task: str) -> str:
        return quote_agent.run(task)

    def ask_sales_agent(task: str) -> str:
        return sales_agent.run(task)

    orchestrator = Agent(
        name="OrchestratorAgent",
        system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
        tools=ORCHESTRATOR_TOOLS_SCHEMA,
        tool_registry={
            "ask_inventory_agent": ask_inventory_agent,
            "ask_quote_agent":     ask_quote_agent,
            "ask_sales_agent":     ask_sales_agent,
        },
        max_iterations=12,
    )
    return orchestrator


# Run your test scenarios by writing them here. Make sure to keep track of them.

def run_test_scenarios():
    
    print("Initializing Database...")
    init_database(db_engine)   # fixed: pass the global db_engine
    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return

    # Get initial state
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    ############
    ############
    ############
    # INITIALIZE YOUR MULTI AGENT SYSTEM HERE
    ############
    ############
    ############

    print("\nInitialising agent system...")
    orchestrator = create_agent_system()
    print("All agents ready.\n")

    results = []
    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")

        print(f"\n=== Request {idx+1} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        # Process request
        request_with_date = (
            f"{row['request']}\n\n"
            f"[Context — Date: {request_date} | Role: {row['job']} | "
            f"Event: {row['event']} | Expected size: {row['need_size']}]"
        )

        ############
        ############
        ############
        # USE YOUR MULTI AGENT SYSTEM TO HANDLE THE REQUEST
        ############
        ############
        ############

        try:
            response = orchestrator.run(request_with_date)
        except Exception as e:
            response = (f"We apologise — an error occurred processing your request. "
                        f"Our team has been notified. (Error: {type(e).__name__})")
            print(f"  ERROR: {e}")

        # Update state
        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print(f"Response: {response}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        results.append(
            {
                "request_id": idx + 1,
                "request_date": request_date,
                "cash_balance": current_cash,
                "inventory_value": current_inventory,
                "response": response,
            }
        )

        time.sleep(1)

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    # Save results
    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    return results


if __name__ == "__main__":
    results = run_test_scenarios()