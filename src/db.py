import sqlite3
from typing import List, Dict, Any
from schema import get_schema
import json
import requests
from typing import Optional, List, Dict, Any
from datetime import datetime
URL = "https://arbml.github.io/masader-webservice/datasets"
class DatasetsDatabase:
    def __init__(self, db_name: str = "datasets.db"):
        """Initialize the database connection and create table if needed."""
        self.db_name = db_name
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.dataset = requests.get(URL).json()
        self.schema = json.loads(get_schema("ar").schema())
        self.keys = [key.replace(" ", "_") for key in self.schema.keys()]
        self.create_table()
        self.create_table()
        self.create_query_log_table()
        self.insert_sample_data()
    
    def create_table(self):
        """Create the datasets table with proper column names."""
        command = "CREATE TABLE IF NOT EXISTS DATASETS ("
        t_mapping = {
            "str": "TEXT",
            "longstr": "TEXT",
            "url": 'TEXT',
            "int": "INTEGER",
            "year": "INTEGER",
            "float": "REAL",
            "bool": "BOOLEAN",
            "list[str]": "TEXT",
            "list[dict[Name, Volume, Unit, Dialect]]": "TEXT"
        }
        for key in self.keys:
            t = self.schema[key]["answer_type"]
            row = f"{key} {t_mapping[t]}"
            if t in ["str", "longstr", "url"]:
                row += " NOT NULL DEFAULT ''"
            elif t in ["int", "year", "float"]:
                row += " NOT NULL DEFAULT 0"
            elif t == "bool":
                row += " NOT NULL DEFAULT FALSE"
            else:
                row += " NOT NULL DEFAULT ''"
            command += row + ", "
        command += "id INTEGER PRIMARY KEY)"
        self.cursor.execute(command)
        self.conn.commit()
    
    def create_query_log_table(self):
        """Create a table to store user queries."""
        command = """
        CREATE TABLE IF NOT EXISTS query_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            natural_language_query TEXT,
            sql_query TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            response_count INTEGER DEFAULT 0
        )
        """
        self.cursor.execute(command)
        self.conn.commit()
        
    def log_query(self, natural_language_query: str, sql_query: str, response_count: int = 0) -> int:
        """Log a user query to the database.
        
        Args:
            natural_language_query: The user's original natural language query
            sql_query: The generated SQL query
            response_count: Number of results returned for this query
            
        Returns:
            int: The ID of the inserted log entry
        """
        self.cursor.execute(
            """
            INSERT INTO query_logs (natural_language_query, sql_query, timestamp, response_count)
            VALUES (?, ?, DATETIME('now'), ?)
            """,
            (natural_language_query, sql_query, response_count)
        )
        self.conn.commit()
        return self.cursor.lastrowid
        
    def get_query_stats(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get query statistics from the log.
        
        Args:
            limit: Maximum number of recent queries to return
            
        Returns:
            List of query log entries with their statistics
        """
        self.cursor.execute(
            """
            SELECT 
                query_text,
                COUNT(*) as query_count,
                AVG(response_count) as avg_results,
                MIN(timestamp) as first_seen,
                MAX(timestamp) as last_seen
            FROM query_logs
            GROUP BY query_text
            ORDER BY query_count DESC
            LIMIT ?
            """,
            (limit,)
        )
        
        columns = [desc[0] for desc in self.cursor.description]
        return [dict(zip(columns, row)) for row in self.cursor.fetchall()]
        
    def get_recent_queries(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get the most recent queries.
        
        Args:
            limit: Maximum number of recent queries to return
            
        Returns:
            List of recent query log entries
        """
        self.cursor.execute(
            """
            SELECT 
                id,
                natural_language_query,
                sql_query,
                timestamp,
                response_count
            FROM query_logs
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,)
        )
        
        columns = [desc[0] for desc in self.cursor.description]
        return [dict(zip(columns, row)) for row in self.cursor.fetchall()]

    def insert_sample_data(self):
        """Insert sample data into the datasets table."""
        # Check if data already exists
        self.cursor.execute("SELECT COUNT(*) FROM DATASETS")
        if self.cursor.fetchone()[0] > 0:
            return
        
        sample_data = []
        for item in self.dataset:
            sample = []
            for key in self.keys:
                value = item[key.replace("_", " ")]
                if isinstance(value, list):
                    if len(value) > 0:
                        if isinstance(value[0], dict):
                            value = json.dumps(value)
                            # print(value)
                        elif isinstance(value[0], str):
                            value = "|".join(value)
                        else:
                            raise ValueError(f"Unsupported list type: {type(value[0])}")
                    else:
                        value = ""
                elif self.schema[key]["answer_type"] == "bool":
                    value = 1 if value == "True" else 0
                sample.append(value)
            sample_data.append(tuple(sample))

        placeholders = ", ".join(["?" for _ in self.keys])
        self.cursor.executemany(
            f"INSERT INTO DATASETS ({', '.join(self.keys)}) VALUES ({placeholders})",
            sample_data
        )
        self.conn.commit()
    
    def get_all_datasets(self) -> List[Dict[str, Any]]:
        """Retrieve all datasets."""
        self.cursor.execute("SELECT * FROM DATASETS")
        columns = [desc[0] for desc in self.cursor.description]
        return [dict(zip(columns, row)) for row in self.cursor.fetchall()]
    
    def get_datasets_by_year(self, year: int) -> List[Dict[str, Any]]:
        """Retrieve DATASETS from a specific year."""
        self.cursor.execute("SELECT * FROM DATASETS WHERE Year = ?", (year,))
        columns = [desc[0] for desc in self.cursor.description]
        return [dict(zip(columns, row)) for row in self.cursor.fetchall()]
    
    def get_datasets_by_volume_range(self, min_volume: float, max_volume: float) -> List[Dict[str, Any]]:
        """Retrieve DATASETS within a volume range."""
        self.cursor.execute(
            "SELECT * FROM DATASETS WHERE Volume BETWEEN ? AND ?",
            (min_volume, max_volume)
        )
        columns = [desc[0] for desc in self.cursor.description]
        return [dict(zip(columns, row)) for row in self.cursor.fetchall()]
    
    def get_datasets_by_name_pattern(self, pattern: str) -> List[Dict[str, Any]]:
        """Search for DATASETS by name pattern."""
        self.cursor.execute(
            "SELECT * FROM DATASETS WHERE Name LIKE ?",
            (f"%{pattern}%",)
        )
        columns = [desc[0] for desc in self.cursor.description]
        return [dict(zip(columns, row)) for row in self.cursor.fetchall()]
    
    def get_datasets_sorted_by_volume(self, descending: bool = True) -> List[Dict[str, Any]]:
        """Retrieve DATASETS sorted by volume."""
        order = "DESC" if descending else "ASC"
        self.cursor.execute(f"SELECT * FROM DATASETS ORDER BY Volume {order}")
        columns = [desc[0] for desc in self.cursor.description]
        return [dict(zip(columns, row)) for row in self.cursor.fetchall()]
    
    def get_datasets_sorted_by_year(self, descending: bool = True) -> List[Dict[str, Any]]:
        """Retrieve DATASETS sorted by year."""
        order = "DESC" if descending else "ASC"
        self.cursor.execute(f"SELECT * FROM DATASETS ORDER BY Year {order}")
        columns = [desc[0] for desc in self.cursor.description]
        return [dict(zip(columns, row)) for row in self.cursor.fetchall()]
    
    def count_datasets_by_year(self) -> List[Dict[str, Any]]:
        """Count DATASETS by year."""
        self.cursor.execute("""
            SELECT Year, COUNT(*) as ProductCount 
            FROM DATASETS 
            GROUP BY Year 
            ORDER BY Year DESC
        """)
        columns = [desc[0] for desc in self.cursor.description]
        return [dict(zip(columns, row)) for row in self.cursor.fetchall()]
    
    def get_volume_statistics(self) -> Dict[str, float]:
        """Get volume statistics (min, max, average)."""
        self.cursor.execute("""
            SELECT 
                MIN(Volume) as MinVolume,
                MAX(Volume) as MaxVolume,
                AVG(Volume) as AvgVolume
            FROM DATASETS
        """)
        result = self.cursor.fetchone()
        return {
            'MinVolume': result[0],
            'MaxVolume': result[1],
            'AvgVolume': result[2]
        }
    
    def get_datasets_above_average_volume(self) -> List[Dict[str, Any]]:
        """Get DATASETS with volume above average."""
        self.cursor.execute("""
            SELECT * FROM DATASETS 
            WHERE Volume > (SELECT AVG(Volume) FROM DATASETS)
        """)
        columns = [desc[0] for desc in self.cursor.description]
        return [dict(zip(columns, row)) for row in self.cursor.fetchall()]
    
    def complex_query(self, year: int, min_volume: float) -> List[Dict[str, Any]]:
        """Complex query with multiple conditions."""
        self.cursor.execute("""
            SELECT * FROM DATASETS 
            WHERE Year >= ? AND Volume > ?
            ORDER BY Volume DESC
        """, (year, min_volume))
        columns = [desc[0] for desc in self.cursor.description]
        return [dict(zip(columns, row)) for row in self.cursor.fetchall()]
    
    def query(self, query: str) -> List[Dict[str, Any]]:
        """Query the database."""
        try:
            self.cursor.execute(query)
            columns = [desc[0] for desc in self.cursor.description]
            return [dict(zip(columns, row)) for row in self.cursor.fetchall()]
        except sqlite3.OperationalError as e:
            print(f"Error executing query: {e}")
            return []
    
    def close(self):
        """Close the database connection."""
        self.conn.close()
