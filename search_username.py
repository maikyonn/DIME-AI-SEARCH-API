
#!/usr/bin/env python3
"""
Username Search Script for LanceDB Dataset
Searches for a specific username in the influencer profiles dataset
"""

import os
import sys
import argparse
from typing import List, Optional
import lancedb
import pandas as pd

from app.config import settings, _resolve_default_db_path


def connect_to_database(db_path: str = None) -> lancedb.DBConnection:
    """Connect to the LanceDB database"""
    if not db_path:
        db_path = settings.DB_PATH or _resolve_default_db_path()

    if not os.path.exists(db_path):
        fallback_path = _resolve_default_db_path()
        if fallback_path and fallback_path != db_path:
            db_path = fallback_path
    
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at: {db_path}")
    
    return lancedb.connect(db_path)


def search_username(username: str, db_path: str = None, table_name: str = "influencer_facets") -> List[dict]:
    """
    Search for a username in the LanceDB dataset
    
    Args:
        username: The username to search for
        db_path: Path to the LanceDB database
        table_name: Name of the table to search in
    
    Returns:
        List of matching records
    """
    try:
        # Connect to database
        db = connect_to_database(db_path)
        
        # Get the table
        if table_name not in db.table_names():
            print(f"Available tables: {db.table_names()}")
            raise ValueError(f"Table '{table_name}' not found in database")
        
        table = db.open_table(table_name)

        base_condition = "content_type = 'profile'"
        lowered = username.lower().replace("'", "''")

        exact_query = f"{base_condition} AND LOWER(username) = '{lowered}'"
        results = table.search().where(exact_query).to_list()

        if not results:
            partial_query = f"{base_condition} AND LOWER(username) LIKE '%{lowered}%'"
            results = table.search().where(partial_query).to_list()
        
        return results
        
    except Exception as e:
        print(f"Error searching for username: {e}")
        return []


def format_result(result: dict) -> str:
    """Format a search result for display"""
    output = []
    output.append(f"Account: {result.get('account') or result.get('username', 'N/A')}")
    output.append(f"Profile Name: {result.get('profile_name') or result.get('display_name', 'N/A')}")
    output.append(f"Followers: {result.get('followers_formatted') or result.get('followers', 'N/A')}")
    output.append(f"Business Category: {result.get('business_category_name') or result.get('occupation', 'N/A')}")
    output.append(f"Business Address: {result.get('business_address') or result.get('location', 'N/A')}")
    output.append(f"Biography: {(result.get('biography') or '')[:100]}...")
    
    # Add engagement metrics if available
    if 'avg_engagement' in result:
        output.append(f"Avg Engagement: {result['avg_engagement']:.2f}")
    
    # Add profile image link if available
    if result.get('profile_image_link') or result.get('profile_image_url'):
        output.append(f"Profile Image: {result.get('profile_image_link') or result.get('profile_image_url')}")
    
    return "\n".join(output)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Search for a username in LanceDB dataset")
    parser.add_argument("username", help="Username to search for")
    parser.add_argument("--db-path", help="Path to LanceDB database")
    parser.add_argument("--table", default="influencer_facets", help="Table name (default: influencer_facets)")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    parser.add_argument("--limit", type=int, default=10, help="Limit number of results (default: 10)")
    
    args = parser.parse_args()
    
    print(f"Searching for username: '{args.username}'")
    
    # Search for the username
    results = search_username(args.username, args.db_path, args.table)
    
    if not results:
        print("No results found.")
        sys.exit(1)
    
    print(f"\nFound {len(results)} result(s):")
    print("=" * 50)
    
    # Limit results if specified
    if args.limit and len(results) > args.limit:
        results = results[:args.limit]
        print(f"Showing first {args.limit} results:")
    
    # Output results
    if args.json:
        import json
        print(json.dumps(results, indent=2, default=str))
    else:
        for i, result in enumerate(results, 1):
            print(f"\nResult {i}:")
            print("-" * 30)
            print(format_result(result))


if __name__ == "__main__":
    main()
