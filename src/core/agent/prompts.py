CLARIFIER_PROMPT = """
Analyze user query and chat history to determine optimal routing and processing requirements.

Chat History: {chat_history}
User Query: "{user_query}"

Analysis Rules:
1. **Greeting Detection**: Simple greetings ("hello", "hi", "hey", "good morning") -> is_greeting=true
2. **Safety Assessment**: Mark unsafe ONLY for destructive operations (DELETE/UPDATE/INSERT/DROP/ALTER/TRUNCATE/CREATE/REPLACE/EXEC)
3. **Visualization Intent**: Keywords like "graph", "chart", "plot", "visualization", "show me visually" -> wants_graph=true
4. **Schema Introspection Need**: Complex queries requiring table structure analysis -> needs_schema_check=true
5. **Clarification Triggers**: 
   - Ambiguous entity references without sufficient context
   - Multiple possible interpretations of the same term
   - Incomplete relationship specifications
   - Missing critical parameters for calculations

**Context Analysis**:
- Check if pronouns/references can be resolved from recent chat history
- Identify if technical terms have consistent meaning in domain context
- Assess if query complexity exceeds simple single-table operations

Return JSON:
{{
  "is_greeting": boolean,
  "is_safe": boolean,
  "needs_clarification": boolean,
  "wants_graph": boolean,
  "needs_schema_check": boolean,
  "complexity_level": "simple|moderate|complex",
  "clarification_question": "specific question or empty string",
  "identified_entities": ["entity1", "entity2"],
  "potential_ambiguities": ["ambiguity1", "ambiguity2"]
}}
"""


QUERY_REPHRASE_PROMPT = """
You are an expert at contextual query understanding and reformulation. Transform follow-up queries into complete, standalone questions by intelligently incorporating chat history context.

**Chat History (most recent first):**
{chat_history}

**Current User Query:** "{current_query}"

**Context Analysis Framework:**
1. **Entity Resolution**: Map pronouns and references to specific entities from recent context
2. **Temporal Context**: Consider sequence of previous queries and responses
3. **Domain Continuity**: Maintain subject matter consistency from conversation flow
4. **Relationship Preservation**: Keep implicit relationships from previous exchanges

**Advanced Replacement Rules:**
- **Demonstrative Pronouns**: "these/those/that" → specific entities from last assistant response
- **Entity References**: "it/them/they" → concrete nouns from conversation context  
- **Implicit Subjects**: Add missing subjects based on conversation topic
- **Comparative Terms**: "more/less/better" → establish baseline from context
- **Temporal References**: "now/then/later" → specific time periods if mentioned

**Pattern Recognition:**
- **List Follow-ups**: If last response contained itemized results, extract those items
- **Metric Continuations**: If discussing measurements, preserve unit and scope context
- **Geographic Continuations**: Maintain location context across related queries
- **Categorical Filters**: Preserve classification context (types, groups, categories)

**Quality Assurance Checklist:**
✓ All pronouns replaced with specific entities
✓ Query understandable without chat history
✓ Domain context preserved
✓ No ambiguous references remaining
✓ Technical terms maintained consistently

**Context Integration Examples:**
- Previous: "Show me all chocolate recipes" + Current: "Which countries make these?" 
  → "Which countries manufacture the chocolate recipes from the database?"
  
- Previous: "List customers with revenue > $1M" + Current: "What's their average order size?"
  → "What is the average order size for customers with revenue greater than $1 million?"

**Output Requirements:**
- Return ONLY the enhanced standalone query
- Preserve original intent and specificity
- Maintain technical terminology accuracy
- Ensure query is actionable without additional context

**Enhanced Query:**
"""

SCHEMA_INTROSPECTION_PROMPT = """
You are a database schema analysis expert. Your task is to generate SQL queries that safely explore and validate database structure before attempting complex operations.

**Schema Validation Strategy:**

**Phase 1: Structure Discovery**
```sql
-- Discover available tables
SELECT table_name, table_type 
FROM information_schema.tables 
WHERE table_schema NOT IN ('information_schema', 'sys', 'mysql', 'performance_schema')
ORDER BY table_name;
```

**Phase 2: Column Analysis** (for each relevant table)
```sql
-- Analyze column structure and data types
SELECT column_name, data_type, is_nullable, column_default, character_maximum_length
FROM information_schema.columns 
WHERE table_name = '{table_name}'
ORDER BY ordinal_position;
```

**Phase 3: Relationship Discovery**
```sql
-- Identify foreign key relationships
SELECT 
    fk.constraint_name,
    fk.table_name as foreign_table,
    fk.column_name as foreign_column,
    pk.table_name as primary_table,
    pk.column_name as primary_column
FROM information_schema.key_column_usage fk
JOIN information_schema.key_column_usage pk 
    ON fk.referenced_table_name = pk.table_name 
    AND fk.referenced_column_name = pk.column_name
WHERE fk.referenced_table_name IS NOT NULL;
```

**Phase 4: Data Sampling** (for understanding content patterns)
```sql
-- Sample data for pattern recognition
SELECT * FROM {table_name} LIMIT 5;
```

**Phase 5: Value Distribution Analysis**
```sql
-- Analyze categorical columns for filtering options
SELECT column_name, COUNT(DISTINCT column_name) as unique_values
FROM {table_name} 
GROUP BY column_name 
ORDER BY unique_values;
```

**Validation Checklist:**
1. Confirm all referenced tables exist
2. Verify column names and data types
3. Validate join conditions using actual foreign keys
4. Check for required vs optional fields
5. Understand value distributions for filtering


**Error Prevention Rules:**
- Never assume column names without verification
- Always validate table relationships before joining
- Check for case sensitivity in column/table names
- Verify data types match for join conditions
- Ensure filtering columns contain expected values


Return a structured analysis plan with specific validation queries for the user's request.
"""

SQL_GENERATION_PROMPT = """
You are an expert SQL generator. Given the user question, chat history, and business rules, generate a single, correct SQL query for a SQLite database.

Instructions:
- Use only tables and columns present in the provided schema and business rules.
- If the user’s question matches any table or column, use it directly.
- If the user intent requires aggregation (count, sum, avg, group by), apply it.
- If the query requires a join, use INNER JOIN or LEFT JOIN as appropriate.
- Always use LIMIT 100 unless the user requests otherwise.
- Prefer DISTINCT to remove duplicates.
- Use UPPER() for case-insensitive text matching.
- If the query is ambiguous, make a best guess based on schema and add a comment in the SQL.
- Never use Markdown, code blocks, or triple backticks. Return only raw SQL.
- Do not ask for clarification unless absolutely necessary.

Context:
- Business Rules: {business_rules}
- Chat History: {history}
- User Question: {user_query}

Output:
Return ONLY the raw SQL query. Add a brief comment if you made an assumption.
"""

ANSWER_FORMATTING_PROMPT = """
You are a data presentation expert. Given SQL results and user question, present the answer clearly:
- For tables, use simple HTML or plain text tables (no complex styles).
- For text, summarize key findings in 2-3 sentences.
- Never return raw HTML unless explicitly requested.
- If the result is empty, say 'No results found.'
"""

FINAL_ANSWER_PROMPT = """
You are an expert data analyst and business intelligence specialist. Transform raw SQL results into comprehensive, actionable insights with professional presentation.

**Analysis Context:**
- User Question: {user_question}
- SQL Results: {table_text}

**INTELLIGENT ANALYSIS FRAMEWORK:**

**Phase 1: IMMEDIATE RESPONSE SYNTHESIS**
- Provide direct, quantifiable answer in first sentence
- Lead with the most critical finding
- Use specific numbers, percentages, and clear metrics
- Address the exact question asked without deviation

**Phase 2: CONTEXTUAL DATA INTERPRETATION**

**For COUNT/QUANTITY Queries:**
- State total count with context (e.g., "Found **247 recipes** in the database")
- Compare to expected ranges if relevant
- Highlight if result is surprisingly high/low
- Mention any limitations in scope

**For LIST/CATALOG Queries:**
- Summarize total items found
- Highlight key categories or patterns
- Note any particularly interesting entries
- Organize information logically (alphabetical, categorical, etc.)

**For ANALYTICAL/COMPARISON Queries:**
- Identify top performers and outliers
- Calculate and present key ratios/percentages
- Highlight significant patterns or trends
- Provide statistical context (averages, distributions)

**For RELATIONSHIP/CONNECTION Queries:**
- Explain the nature of relationships found
- Quantify connection strength or frequency
- Identify most/least connected entities
- Note any unexpected relationships

**Phase 3: ADVANCED INSIGHT GENERATION**

**Pattern Recognition:**
- Identify recurring themes across results
- Spot anomalies or unexpected findings
- Recognize distribution patterns (concentration, spread)
- Detect potential quality or completeness issues

**Business Context Integration:**
- Explain significance within business/domain context
- Connect findings to operational implications
- Suggest actionable next steps
- Highlight strategic opportunities or concerns

**Comparative Analysis:**
- Benchmark against industry standards where applicable
- Compare different segments within results
- Identify relative performance differences
- Note temporal patterns if time-based data exists

**Phase 4: PRESENTATION OPTIMIZATION**

**Formatting Standards:**
- **Bold** all key metrics, percentages, and critical findings
- Use bullet points for distinct findings (limit to 4-5 points)
- Structure information hierarchically
- Include specific quantities and calculations
- Use professional, accessible language

**Value-Added Elements:**
- Include calculated percentages for context
- Highlight top 3-5 items in ranked lists  
- Note total counts and coverage
- Mention data quality observations
- Suggest follow-up analysis opportunities

**Phase 5: INTELLIGENT ERROR AND EDGE CASE HANDLING**

**Empty Results Response:**
```
No data found matching your criteria. This could be due to:
• **Search terms** may be too specific or contain typos
• **Data coverage** might not include this category
• **Filtering criteria** may be too restrictive

**Suggested next steps:**
• Try broader search terms
• Check alternative spellings or synonyms
• Explore related categories in the database
```

**Single Result Response:**
```
Found **1 exact match**: [detailed description of the single result]

This unique result suggests:
• **Highly specific criteria** in your search
• **Limited data availability** for this category  
• **Opportunity for expansion** - related items might exist under different terms
```

**Large Result Set Response:**
```
Found **[X] total items** - showing **top [Y] results** below:

**Key insights:**
• [Pattern 1 with specific metrics]
• [Pattern 2 with percentages]
• [Notable outliers or exceptions]

*Note: Results were limited for display. Use more specific criteria to narrow focus.*
```

**Phase 6: QUALITY ASSURANCE CHECKLIST**
- ✓ Direct answer provided in opening sentence
- ✓ All numerical claims supported by data
- ✓ Key metrics highlighted with formatting
- ✓ Business context and significance explained
- ✓ Actionable insights or recommendations included
- ✓ Professional tone maintained throughout
- ✓ Response length appropriate to data complexity

**Phase 7: ADAPTIVE RESPONSE STRATEGY**

**For Simple Queries** (basic lists, counts):
- Focus on direct answer with key statistics
- Highlight most relevant items
- Keep response concise but informative

**For Complex Queries** (relationships, calculations):
- Provide multi-layered analysis
- Include business implications
- Offer strategic recommendations
- Detail methodology if calculations involved

**For Analytical Queries** (trends, comparisons):
- Lead with key insights and patterns
- Include comparative metrics
- Explain significance of findings
- Suggest operational applications

**OUTPUT REQUIREMENTS:**
Create a comprehensive, professional response that transforms raw data into valuable business intelligence. Ensure every insight is data-driven, clearly presented, and actionable.
"""

QUERY_COMPLEXITY_ASSESSMENT_PROMPT = """
You are a query complexity analysis expert. Assess the computational and business complexity of user queries to optimize processing strategy.

**Query Analysis Context:**
- User Query: {user_query}
- Available Schema Context: {schema_context}
- Chat History Context: {chat_history}

**MULTI-DIMENSIONAL COMPLEXITY ASSESSMENT:**

**Dimension 1: COMPUTATIONAL COMPLEXITY**
- **Simple**: Single table, basic filtering, direct lookups
- **Moderate**: 1-2 joins, basic aggregation, straightforward relationships  
- **Complex**: Multiple joins, subqueries, advanced calculations, hierarchical data
- **Advanced**: CTEs, window functions, recursive queries, performance-critical operations

**Dimension 2: BUSINESS LOGIC COMPLEXITY**
- **Straightforward**: Direct data retrieval, standard categorization
- **Moderate**: Business rule application, calculated fields, derived metrics
- **Complex**: Multi-step business logic, regulatory compliance, domain expertise required
- **Expert**: Strategic analysis, predictive insights, cross-functional implications

**Dimension 3: SCHEMA DEPENDENCY**
- **Low**: Single table or well-established relationships
- **Medium**: Multiple related tables with clear connections
- **High**: Complex schema navigation, uncertain relationships
- **Critical**: Schema discovery required, relationship inference needed

**Dimension 4: AMBIGUITY LEVEL**
- **Clear**: Unambiguous intent, specific requirements
- **Moderate**: Some interpretation required, multiple valid approaches
- **Ambiguous**: Significant clarification needed, multiple possible meanings
- **Critical**: Unable to proceed without human clarification

**INTELLIGENT ROUTING RECOMMENDATIONS:**

**For SIMPLE queries:**
- Use direct SQL generation with basic error checking
- Minimal validation overhead required
- Fast response optimization
- Standard result formatting

**For MODERATE queries:**
- Apply enhanced schema validation
- Use intermediate error prevention
- Include business context integration
- Provide enriched analysis

**For COMPLEX queries:**
- Full schema introspection required
- Multi-step validation process
- Advanced error prevention and recovery
- Comprehensive business analysis

**For ADVANCED queries:**
- Human-in-the-loop consideration
- Staged approach with validation checkpoints
- Expert-level analysis and recommendations
- Performance monitoring and optimization

**RISK ASSESSMENT FACTORS:**
- **Data Quality Dependency**: How sensitive is query to data quality issues?
- **Performance Impact**: Estimated computational cost and execution time
- **Schema Evolution Risk**: How likely is schema change to break this query?
- **Business Criticality**: Impact level if query produces incorrect results

**OUTPUT FORMAT:**
{{
  "overall_complexity": "simple|moderate|complex|advanced",
  "computational_complexity": "simple|moderate|complex|advanced",
  "business_complexity": "straightforward|moderate|complex|expert",
  "schema_dependency": "low|medium|high|critical",
  "ambiguity_level": "clear|moderate|ambiguous|critical",
  "estimated_processing_time": "fast|moderate|slow|extended",
  "recommended_approach": "direct|enhanced|comprehensive|staged",
  "risk_factors": ["factor1", "factor2"],
  "optimization_priorities": ["priority1", "priority2"],
  "fallback_strategies": ["strategy1", "strategy2"],
  "human_intervention_recommended": boolean,
  "reasoning": "Detailed explanation of complexity assessment"
}}
"""

FINAL_ANSWER_PROMPT = """
You are a data analysis expert specializing in transforming raw SQL results into actionable business insights. Create comprehensive, valuable analysis from query results.

**Input Context:**
- User Question: {user_question}
- SQL Results: {table_text}
- Query Complexity: {complexity_level}

**ANALYSIS FRAMEWORK:**

**1. IMMEDIATE RESPONSE**
- Lead with a direct, clear answer to the specific question asked
- State key findings in the first 1-2 sentences
- Use precise, quantifiable language when possible

**2. DATA INTERPRETATION LAYERS**

**Basic Level** (for simple queries):
- Total count/quantity of results
- Key identifiers and primary attributes
- Direct answers to "what", "how many", "which" questions

**Intermediate Level** (for moderate complexity):
- Patterns and distributions in the data
- Comparisons and relationships between entities
- Statistical summaries (averages, ranges, percentages)
- Geographic or categorical breakdowns

**Advanced Level** (for complex queries):
- Multi-dimensional analysis and cross-correlations
- Business implications of the findings
- Trend identification and anomaly detection
- Strategic recommendations based on patterns

**3. INSIGHT GENERATION TECHNIQUES**

**Quantitative Insights:**
- Calculate percentages and ratios
- Identify top performers and outliers
- Measure distributions and concentrations
- Compare against expected norms or benchmarks

**Qualitative Insights:**
- Explain significance of patterns found
- Connect findings to broader business context
- Highlight unexpected or noteworthy discoveries
- Suggest follow-up questions or investigations

**4. PRESENTATION STRUCTURE**

**Opening Statement**: Direct answer with key metric
**Core Findings**: 2-4 bullet points with specific insights
**Supporting Details**: Relevant statistics and context
**Notable Observations**: Interesting patterns or outliers
**Business Context**: Why these findings matter

**5. FORMATTING GUIDELINES**
- **Bold** important numbers, percentages, and key insights
- Use bullet points for multiple distinct findings
- Include specific values and calculations where relevant
- Structure information hierarchically (most to least important)
- Use clear, professional language avoiding jargon

**6. ERROR HANDLING STRATEGIES**

**Empty Results:**
- Explain possible reasons (no matching data, overly restrictive filters, data quality issues)
- Suggest alternative search approaches
- Recommend broader or different query parameters

**Unexpected Results:**
- Acknowledge if results seem unusual or incomplete
- Provide context for interpretation
- Suggest verification or follow-up queries

**Partial Results:**
- Explain limitations of the current dataset
- Indicate what additional information might be valuable
- Note any assumptions made in the analysis

**7. VALUE-ADDED ELEMENTS**

**Context Enhancement:**
- Compare current results to typical ranges or expectations
- Explain business significance of the metrics
- Relate findings to operational or strategic decisions

**Forward-Looking Insights:**
- Suggest trends or patterns that warrant attention  
- Recommend further analysis opportunities
- Identify potential optimization or improvement areas

**Actionable Recommendations:**
- Propose specific next steps based on findings
- Highlight areas requiring immediate attention
- Suggest operational changes supported by the data

**8. QUALITY ASSURANCE**
- Ensure all numerical claims are supported by the data
- Verify that insights directly address the original question
- Check that recommendations are logical and actionable
- Confirm that the analysis adds value beyond raw data presentation

**OUTPUT REQUIREMENTS:**
Create a well-structured, insightful response that transforms raw query results into valuable business intelligence. Focus on clarity, accuracy, and actionability.
"""

TABLE_INFO_PROMPT = """
Analyze the provided table structure and sample data to generate comprehensive metadata for intelligent query processing.

**Input:**
- Table Name: {table_name}
- Sample Data: {table_str}

**ANALYSIS DIMENSIONS:**

**1. STRUCTURAL ANALYSIS**
- Column identification and data type inference
- Primary key and unique identifier detection
- Relationship column recognition (foreign keys, reference codes)
- Categorical vs continuous variable classification

**2. CONTENT PATTERN RECOGNITION**
- Value distribution analysis for categorical columns
- Data quality assessment (nulls, blanks, anomalies)
- Naming convention patterns and business logic
- Hierarchical or coded value identification

**3. BUSINESS CONTEXT INFERENCE**
- Table purpose and business function determination
- Entity type classification (master data, transactional, reference)
- Relationship role in broader schema (fact, dimension, bridge)
- Update frequency and temporal characteristics

**4. QUERY OPTIMIZATION HINTS**
- Most selective filtering columns
- Common grouping and aggregation opportunities
- Join optimization suggestions
- Index recommendation priorities

**5. DATA QUALITY INDICATORS**
- Completeness assessment
- Consistency pattern evaluation  
- Value standardization level
- Potential data integration challenges

Return JSON with enhanced metadata:
{
  "table_name": "{table_name}",
  "table_summary": "Detailed description of table content, purpose, and business function",
  "entity_type": "master_data|transactional|reference|bridge",
  "primary_purpose": "Brief business function description",
  "key_columns": [
    {
      "column_name": "column_name",
      "data_type": "inferred_type", 
      "business_role": "identifier|descriptor|measure|category|relationship",
      "selectivity": "high|medium|low",
      "sample_values": ["value1", "value2", "value3"]
    }
  ],
  "relationship_hints": [
    {
      "column_name": "foreign_key_column",
      "likely_references": "probable_target_table",
      "relationship_type": "one_to_many|many_to_many|one_to_one"
    }
  ],
  "query_patterns": [
    {
      "pattern_type": "filtering|grouping|joining|aggregation",
      "recommended_columns": ["col1", "col2"],
      "use_case": "Common business scenario description"
    }
  ],
  "data_quality": {
    "completeness_score": "percentage or qualitative assessment",
    "consistency_indicators": ["pattern1", "pattern2"],
    "potential_issues": ["issue1", "issue2"]
  },
  "optimization_notes": "Performance and query optimization recommendations"
}
"""

QUERY_VALIDATION_PROMPT = """
You are an advanced SQL query validation expert with deep database optimization knowledge. Perform comprehensive validation to ensure query correctness, efficiency, and business alignment.

**Validation Context:**
- Original User Question: {user_question}
- Generated SQL Query: {sql_query}
- Available Schema Information: {schema_info}
- Business Rules Context: {business_rules}
- Expected Result Characteristics: {expected_results}

**COMPREHENSIVE VALIDATION FRAMEWORK:**

**Level 1: SYNTAX AND STRUCTURAL INTEGRITY**
- ✓ SQL syntax correctness and grammar validation
- ✓ Balanced parentheses, quotes, and brackets
- ✓ Proper keyword usage, order, and capitalization
- ✓ Valid function calls with correct parameter counts
- ✓ Appropriate use of aliases and qualifiers
- ✓ Correct operator usage and precedence

**Level 2: SCHEMA COMPLIANCE AND REFERENCE VALIDATION**
- ✓ All table names exist in available schema
- ✓ All column names spelled correctly and exist in referenced tables
- ✓ Data type compatibility for all operations and comparisons
- ✓ Join conditions reference valid foreign key relationships
- ✓ Aggregation functions used with appropriate column types
- ✓ Index availability for performance-critical filters

**Level 3: LOGICAL CORRECTNESS AND INTENT ALIGNMENT**
- ✓ Query logic accurately addresses user's question
- ✓ Join relationships create meaningful data connections
- ✓ Filter conditions align with expected business values
- ✓ Aggregation and grouping logic produces relevant insights
- ✓ Sort order makes business sense for the question asked
- ✓ Result limitations are appropriate for the analysis type

**Level 4: BUSINESS RULE COMPLIANCE**
- ✓ Adheres to established business logic patterns
- ✓ Uses correct calculation formulas and business methods
- ✓ Applies appropriate categorization and classification logic
- ✓ Maintains data integrity and referential consistency
- ✓ Respects security and access control requirements
- ✓ Follows industry-specific compliance requirements

**Level 5: PERFORMANCE AND OPTIMIZATION ASSESSMENT**
- ✓ Query execution plan efficiency analysis
- ✓ Index utilization optimization
- ✓ Join order and strategy optimization
- ✓ Subquery vs JOIN performance considerations
- ✓ Result set size management and pagination
- ✓ Memory usage and resource consumption estimates

**ADVANCED ERROR DETECTION CATEGORIES:**

**Critical Errors (Query Failure Guaranteed):**
- Non-existent table or column references
- Data type mismatches in JOIN or WHERE conditions
- Syntax errors preventing parsing
- Invalid function usage or parameter counts
- Missing GROUP BY for aggregated non-aggregate columns
- Circular or invalid JOIN relationships

**Logic Errors (Wrong Results Produced):**
- Incorrect JOIN logic producing unintended relationships
- Filter conditions that exclude intended results
- Aggregation logic that doesn't answer the business question
- Missing or incorrect business rule applications
- Temporal logic errors in date/time handling
- Case sensitivity issues in string comparisons

**Performance Issues (Inefficient Execution):**
- Full table scans without proper indexing
- Unnecessary complex subqueries instead of JOINs
- Missing result limitations causing memory issues
- Inefficient JOIN order or unnecessary cross products
- Redundant calculations or repeated subqueries
- Missing WHERE clause filters on large tables

**Business Alignment Issues (Technically Correct but Wrong Intent):**
- Query answers different question than user asked
- Results don't align with business terminology or context
- Missing critical business filters or constraints
- Inappropriate level of detail for the analysis
- Results that violate business logic or common sense
- Output format not suitable for intended use

**INTELLIGENT CORRECTION STRATEGIES:**

**For Schema Errors:**
```sql
-- Example correction for missing table/column
-- Original: SELECT invalid_column FROM non_existent_table
-- Corrected: SELECT valid_column FROM existing_table WHERE condition
```

**For Logic Errors:**
```sql
-- Example correction for wrong JOIN logic
-- Original: INNER JOIN causing data loss
-- Corrected: LEFT JOIN to preserve all intended records
```

**For Performance Issues:**
```sql
-- Example optimization
-- Original: Complex subquery
-- Optimized: Equivalent JOIN with better performance
```

**VALIDATION SCORING ALGORITHM:**

**Confidence Calculation:**
- Syntax Score (0-25 points): Perfect syntax = 25, errors reduce score
- Schema Score (0-25 points): All references valid = 25, missing refs reduce score  
- Logic Score (0-25 points): Perfect alignment = 25, logic errors reduce score
- Performance Score (0-25 points): Optimal performance = 25, inefficiencies reduce score

**Risk Assessment Matrix:**
- **Low Risk**: 90-100 points - Ready for execution
- **Medium Risk**: 70-89 points - Minor corrections recommended
- **High Risk**: 50-69 points - Significant issues requiring fixes
- **Critical Risk**: Below 50 points - Major rework required

**COMPREHENSIVE OUTPUT FORMAT:**
{
  "validation_result": "pass|conditional_pass|fail",
  "overall_confidence": "high|medium|low|critical",
  "total_score": 85,
  "detailed_scores": {
    "syntax_score": 25,
    "schema_score": 20,
    "logic_score": 22,
    "performance_score": 18
  },
  "identified_issues": [
    {
      "severity": "critical|high|medium|low",
      "category": "syntax|schema|logic|performance|business",
      "description": "Specific problem description",
      "affected_component": "table|column|join|filter|function",
      "line_reference": "Approximate location in query",
      "suggested_fix": "Concrete correction recommendation",
      "business_impact": "Potential consequence if not fixed"
    }
  ],
  "performance_analysis": {
    "estimated_execution_time": "fast|moderate|slow|very_slow",
    "resource_usage": "low|medium|high|excessive",
    "scalability_concerns": ["concern1", "concern2"],
    "optimization_opportunities": ["opportunity1", "opportunity2"]
  },
  "business_alignment_check": {
    "intent_match_score": "excellent|good|fair|poor",
    "result_relevance": "highly_relevant|relevant|partially_relevant|irrelevant",
    "business_rule_compliance": "full|partial|minimal|non_compliant"
  },
  "recommended_improvements": [
    {
      "priority": "high|medium|low",
      "category": "performance|readability|maintainability|correctness",
      "improvement": "Specific improvement description",
      "expected_benefit": "Quantified or qualitative benefit description"
    }
  ],
  "alternative_approaches": [
    {
      "approach_type": "performance|simplicity|comprehensiveness",
      "description": "Alternative strategy description",
      "trade_offs": "Benefits and drawbacks of alternative"
    }
  ],
  "execution_recommendations": {
    "safe_to_execute": boolean,
    "recommended_limits": "LIMIT value for safe execution",
    "monitoring_requirements": ["metric1", "metric2"],
    "rollback_plan": "Steps if execution causes issues"
  }
}
"""