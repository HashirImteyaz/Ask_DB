# agent/response_formatter.py

import pandas as pd
from typing import Union
import html

def format_dataframe_as_html_table(df: pd.DataFrame) -> str:
    """Convert DataFrame to a clean, styled HTML table."""
    
    # Create HTML table with styling
    html_table = """
    <div style="overflow-x: auto; margin: 15px 0; max-width: 100%; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
        <table style="border-collapse: collapse; width: 100%; min-width: 500px; font-size: 13px; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
            <thead>
                <tr style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
    """
    
    # Add headers
    for col in df.columns:
        html_table += f'<th style="padding: 12px 8px; text-align: left; font-weight: 600; border: 1px solid #ddd;">{html.escape(str(col))}</th>'
    
    html_table += "</tr></thead><tbody>"
    
    # Add rows with alternating colors
    for idx, row in df.iterrows():
        row_color = "#f8f9ff" if idx % 2 == 0 else "#ffffff"
        html_table += f'<tr style="background-color: {row_color};">'
        
        for col in df.columns:
            value = row[col]
            
            # Format different data types nicely
            if pd.isna(value):
                formatted_value = "<em style='color: #999;'>—</em>"
            elif isinstance(value, (int, float)):
                if isinstance(value, float):
                    formatted_value = f"{value:,.2f}" if value != int(value) else f"{int(value):,}"
                else:
                    formatted_value = f"{value:,}"
            else:
                formatted_value = html.escape(str(value))
            
            # Align numbers to the right, text to the left
            text_align = "right" if isinstance(value, (int, float)) and not pd.isna(value) else "left"
            html_table += f'<td style="padding: 10px 8px; border: 1px solid #eee; text-align: {text_align};">{formatted_value}</td>'
        
        html_table += "</tr>"
    
    html_table += """
            </tbody>
        </table>
    </div>
    """
    
    return html_table

def format_dataframe_response(df: pd.DataFrame, user_question: str) -> str:
    """Enhanced formatter with better insights and analysis."""
    
    # If DataFrame is empty
    if df.empty:
        return "No results found."

    # If single column, single value (COUNT, SUM, etc.)
    if df.shape[1] == 1 and df.shape[0] == 1:
        col = df.columns[0]
        val = df.iloc[0, 0]
        return f"{col}: {val}"

    # For small tables, use plain text table
    if df.shape[0] <= 10 and df.shape[1] <= 5:
        # Plain text table
        header = " | ".join([str(c) for c in df.columns])
        rows = [" | ".join([str(df.iloc[i, j]) for j in range(df.shape[1])]) for i in range(df.shape[0])]
        table = "\n".join([header] + rows)
        return table

    # Otherwise, use simple HTML table (no complex styles)
    return df.to_html(index=False, border=1)
    if len(df) == 1 and len(df.columns) == 1:
        value = df.iloc[0, 0]
        if isinstance(value, (int, float)):
            formatted_value = f"<div style='font-size: 24px; font-weight: bold; color: #2c3e50; text-align: center; padding: 20px;'>{value:,}</div>"
            
            # Add context based on the question
            if "count" in user_question.lower() or "how many" in user_question.lower():
                if value == 0:
                    insight = "<p style='text-align: center; color: #666; margin-top: 10px;'>No matching records found.</p>"
                elif value == 1:
                    insight = "<p style='text-align: center; color: #666; margin-top: 10px;'>Found exactly one matching record.</p>"
                else:
                    insight = f"<p style='text-align: center; color: #666; margin-top: 10px;'>Found {value:,} matching records in total.</p>"
                return formatted_value + insight
            return formatted_value
        else:
            return f"<div style='font-size: 20px; font-weight: bold; color: #2c3e50; text-align: center; padding: 20px;'>{value}</div>"
    
    # Add summary statistics for numeric columns
    summary_html = ""
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0 and len(df) > 1:
        summary_stats = []
        for col in numeric_cols:
            total = df[col].sum()
            avg = df[col].mean()
            max_val = df[col].max()
            min_val = df[col].min()
            
            if total != avg:  # More than one row
                summary_stats.append(f"**{col}**: Total: {total:,.0f}, Average: {avg:,.1f}, Range: {min_val:,.0f} - {max_val:,.0f}")
            else:
                summary_stats.append(f"**{col}**: {total:,.1f}")
        
        if summary_stats:
            summary_html = f"""
            <div style='background-color: #f8f9ff; border-left: 4px solid #667eea; padding: 15px; margin: 10px 0; border-radius: 4px;'>
                <h4 style='margin: 0 0 10px 0; color: #2c3e50;'>📊 Quick Summary</h4>
                {'<br>'.join(summary_stats)}
            </div>
            """
    
    # Small table - show with nice formatting
    if len(df) <= 20:
        html_table = format_dataframe_as_html_table(df)
        return f"{summary_html}{html_table}"
    
    # Large table - show first 10 rows + summary
    else:
        html_table = format_dataframe_as_html_table(df.head(10))
        return f"""
        {summary_html}
        {html_table}
        <div style='text-align: center; margin: 15px 0; padding: 10px; background-color: #f0f2ff; border-radius: 8px; color: #5a6c7d; font-style: italic;'>
            Showing first 10 rows of {len(df):,} total records
        </div>
        """



# # agent/response_formatter.py

# import pandas as pd
# from typing import Union

# def format_dataframe_response(df: pd.DataFrame, user_question: str) -> str:
#     """Simple, direct formatter with minimal fluff."""
    
#     # Single value result (like COUNT queries)
#     if len(df) == 1 and len(df.columns) == 1:
#         value = df.iloc[0, 0]
#         if isinstance(value, (int, float)):
#             return f"**{value:,}**"
#         else:
#             return f"**{value}**"
    
#     # Small table - just show it
#     if len(df) <= 20:
#         try:
#             return df.to_markdown(index=False)
#         except:
#             return df.to_string(index=False)
    
#     # Large table - show first few rows + summary
#     else:
#         try:
#             preview = df.head(10).to_markdown(index=False)
#             return f"{preview}\n\n*({len(df)} total rows)*"
#         except:
#             return f"{df.head(10).to_string(index=False)}\n\n*({len(df)} total rows)*"
