
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


def connect_to_database(db_path: str = None) -> lancedb.DBConnection:
    """Connect to the LanceDB database"""
    if not db_path:
        # Use the same path logic as in config.py
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        db_path = os.path.join(project_root, "DIME-AI-DB", "influencers_vectordb")
    
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at: {db_path}")
    
    return lancedb.connect(db_path)


def search_username(username: str, db_path: str = None, table_name: str = "influencer_profiles") -> List[dict]:
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
        
        # Search for the username
        # Try exact match first
        query = f"account = '{username}'"
        results = table.search().where(query).to_list()
        
        # If no exact match, try case-insensitive partial match
        if not results:
            query = f"LOWER(account) LIKE '%{username.lower()}%'"
            results = table.search().where(query).to_list()
        
        return results
        
    except Exception as e:
        print(f"Error searching for username: {e}")
        return []


def format_result(result: dict) -> str:
    """Format a search result for display"""
    output = []
    output.append(f"Account: {result.get('account', 'N/A')}")
    output.append(f"Profile Name: {result.get('profile_name', 'N/A')}")
    output.append(f"Followers: {result.get('followers_formatted', result.get('followers', 'N/A'))}")
    output.append(f"Business Category: {result.get('business_category_name', 'N/A')}")
    output.append(f"Business Address: {result.get('business_address', 'N/A')}")
    output.append(f"Biography: {result.get('biography', 'N/A')[:100]}...")
    
    # Add engagement metrics if available
    if 'avg_engagement' in result:
        output.append(f"Avg Engagement: {result['avg_engagement']:.2f}")
    
    # Add profile image link if available
    if result.get('profile_image_link'):
        output.append(f"Profile Image: {result['profile_image_link']}")
    
    return "\n".join(output)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Search for a username in LanceDB dataset")
    parser.add_argument("username", help="Username to search for")
    parser.add_argument("--db-path", help="Path to LanceDB database")
    parser.add_argument("--table", default="influencer_profiles", help="Table name (default: influencer_profiles)")
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