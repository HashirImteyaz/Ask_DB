# test_summary.py - Test summary and analysis

import pandas as pd
import json
from pathlib import Path

def analyze_evaluation_questions():
    """Analyze the evaluation questions and provide insights."""
    
    print("="*70)
    print("NLQ SYSTEM TEST ANALYSIS")
    print("="*70)
    
    # Load evaluation questions
    try:
        eval_df = pd.read_excel("evaluation/evaluation_questions.xlsx")
        print(f"Loaded {len(eval_df)} evaluation questions from Excel file")
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return
    
    # Clean data
    eval_df = eval_df.dropna(subset=['Human Question'])
    
    print(f"\nQUESTION BREAKDOWN BY DIFFICULTY:")
    difficulty_counts = eval_df['Difficulty'].value_counts()
    for difficulty, count in difficulty_counts.items():
        print(f"  {difficulty}: {count} questions ({count/len(eval_df)*100:.1f}%)")
    
    print(f"\nSAMPLE QUESTIONS BY DIFFICULTY:")
    
    for difficulty in ["Easy", "Medium", "Hard"]:
        questions = eval_df[eval_df['Difficulty'] == difficulty]
        if not questions.empty:
            print(f"\n{difficulty.upper()} Questions:")
            for idx, row in questions.head(3).iterrows():
                print(f"  • {row['Human Question']}")
    
    print(f"\nSQL COMPLEXITY ANALYSIS:")
    sql_features = {
        'Simple SELECT': 0,
        'With JOIN': 0,
        'With GROUP BY': 0,
        'With aggregation (COUNT/SUM)': 0,
        'With LIKE pattern': 0,
        'With DISTINCT': 0
    }
    
    for sql in eval_df['SQL Query']:
        if pd.notna(sql):
            sql_upper = str(sql).upper()
            if 'SELECT' in sql_upper and 'JOIN' not in sql_upper:
                sql_features['Simple SELECT'] += 1
            if 'JOIN' in sql_upper:
                sql_features['With JOIN'] += 1
            if 'GROUP BY' in sql_upper:
                sql_features['With GROUP BY'] += 1
            if any(agg in sql_upper for agg in ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN']):
                sql_features['With aggregation (COUNT/SUM)'] += 1
            if 'LIKE' in sql_upper:
                sql_features['With LIKE pattern'] += 1
            if 'DISTINCT' in sql_upper:
                sql_features['With DISTINCT'] += 1
    
    for feature, count in sql_features.items():
        print(f"  {feature}: {count} queries")
    
    print(f"\nEXPECTED SYSTEM CAPABILITIES:")
    print("The NLQ system should be able to handle:")
    print("  [+] Basic ingredient and recipe lookups")
    print("  [+] Filtering by business units and countries")
    print("  [+] Join operations between Specifications and RecipeExplosion")
    print("  [+] Aggregations for counting and summing")
    print("  [+] Pattern matching with LIKE operations")
    print("  [+] Business logic understanding (codes, statuses)")
    
    print(f"\nCHALLENGING ASPECTS:")
    print("  - Complex business logic (authorization groups, plant codes)")
    print("  - Multi-table joins with proper filtering")
    print("  - Understanding percentage vs quantity contributions")
    print("  - Handling case-insensitive text matching")
    
    # Show some example expected SQL patterns
    print(f"\nEXAMPLE EXPECTED SQL PATTERNS:")
    
    easy_example = eval_df[eval_df['Difficulty'] == 'Easy'].iloc[0]
    print(f"\nEASY: {easy_example['Human Question']}")
    print(f"Expected SQL: {easy_example['SQL Query']}")
    
    if not eval_df[eval_df['Difficulty'] == 'Hard'].empty:
        hard_example = eval_df[eval_df['Difficulty'] == 'Hard'].iloc[0]
        print(f"\nHARD: {hard_example['Human Question']}")
        print(f"Expected SQL: {hard_example['SQL Query'][:200]}...")
    
    return eval_df

def system_readiness_check():
    """Check if the system components are ready."""
    
    print(f"\n" + "="*70)
    print("SYSTEM READINESS CHECK")
    print("="*70)
    
    components = {
        'Database': 'DATA/plm_updated.db',
        'Schema Description': 'src/config/schema_description.json',
        'Evaluation Questions': 'evaluation/evaluation_questions.xlsx',
        'Agent Code': 'agent/graph.py',
        'Vector System': 'data_processing/vectors.py',
        'Main Application': 'main_app.py'
    }
    
    ready_count = 0
    for component, path in components.items():
        if Path(path).exists():
            print(f"  [+] {component}: Ready")
            ready_count += 1
        else:
            print(f"  [-] {component}: Missing ({path})")
    
    readiness = ready_count / len(components)
    print(f"\nSystem Readiness: {ready_count}/{len(components)} ({readiness:.1%})")
    
    if readiness >= 1.0:
        print("[SUCCESS] System is fully ready for testing!")
    elif readiness >= 0.8:
        print("[WARNING] System is mostly ready - minor issues")
    else:
        print("[ERROR] System has significant missing components")
    
    return readiness >= 0.8

def main():
    """Main analysis function."""
    
    # Analyze questions
    eval_df = analyze_evaluation_questions()
    
    # Check system readiness
    system_ready = system_readiness_check()
    
    if system_ready and eval_df is not None:
        print(f"\n" + "="*70)
        print("RECOMMENDED TESTING APPROACH")
        print("="*70)
        print("1. Start the server: python main_app.py")
        print("2. Run manual test: python manual_test.py")
        print("3. For full evaluation: python run_comprehensive_test.py")
        print("4. Check results in evaluation/ folder")
        
        print(f"\nSUCCESS CRITERIA:")
        print(f"  - Easy questions: >80% success rate expected")
        print(f"  - Medium questions: >60% success rate expected") 
        print(f"  - Hard questions: >40% success rate expected")
        print(f"  - Overall: >60% success rate for good performance")
    
    return True

if __name__ == "__main__":
    main()