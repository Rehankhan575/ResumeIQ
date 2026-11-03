"""
scripts/build_skills_taxonomy.py
================================
Build the unified skills taxonomy for the rule-based extractor in Step 3a.

Sources (in priority order):
  1. data/taxonomies/skills_curated.csv (hand-curated 480+ skills across 16 domains)
  2. O*NET software_skills.csv (workplace examples with >=3 occupations or Hot Tech, <=4 words)
  3. SKILL strings from ner_synthetic (301 distinct, mapped to dominant industry)

Output:
  - data/taxonomies/skills_curated.csv (skill, aliases, category)
  - data/taxonomies/ambiguous_terms.csv (term, rule)
  - data/taxonomies/skills.csv (skill, aliases, category, source)

Privacy: prints counts and skill strings only, never resume body text.
"""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from loguru import logger

# -- Paths ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_TAXONOMIES = ROOT / "data" / "taxonomies"
DATA_TAXONOMIES.mkdir(parents=True, exist_ok=True)

ONET_PATH = DATA_RAW / "onet" / "software_skills.csv"
SYNTH_DIR = DATA_RAW / "ner_synthetic"

CURATED_PATH = DATA_TAXONOMIES / "skills_curated.csv"
AMBIGUOUS_PATH = DATA_TAXONOMIES / "ambiguous_terms.csv"
SKILLS_PATH = DATA_TAXONOMIES / "skills.csv"

ALLOWLIST_SINGLE_CHAR: Set[str] = {"c", "r", "go"}

# Common English words list for collision reporting
COMMON_ENGLISH_WORDS: Set[str] = {
    "go", "r", "c", "java", "swift", "rust", "spark", "excel", "word",
    "access", "sales", "project", "management", "chef", "puppet", "spring",
    "react", "make", "bash", "git", "rest", "dart", "sass", "yarn", "jira",
    "triage", "welding", "baking", "grilling", "roasting", "communication",
    "leadership", "teamwork", "reading", "grading", "planning", "teaching",
    "cleaning", "delivery", "dispatch", "drilling", "nursing", "coaching",
    "training", "writing", "reporting", "supervision", "scheduling", "carpentry",
    "plumbing", "contract", "legal", "audit", "tax", "payroll", "equity",
    "banking", "marketing", "content", "media", "video", "photo", "music",
    "sound", "design", "fashion", "nutrition", "fitness", "wellness", "care",
}


# -- Curated Skills (480+ skills across 16 domains) --------------------------
RAW_CURATED_SKILLS: List[Tuple[str, str, str]] = [
    # 1. Programming Languages
    ("python", "py|python3|python2", "programming_languages"),
    ("javascript", "js|ecmascript|es6|es2015", "programming_languages"),
    ("typescript", "ts", "programming_languages"),
    ("java", "java programming|core java", "programming_languages"),
    ("c++", "cpp|c plus plus", "programming_languages"),
    ("c#", "csharp|c sharp|.net c#", "programming_languages"),
    ("c", "c language|c programming", "programming_languages"),
    ("r", "r language|r programming|r-project", "programming_languages"),
    ("go", "golang", "programming_languages"),
    ("rust", "rust language|rust programming", "programming_languages"),
    ("php", "php7|php8", "programming_languages"),
    ("ruby", "ruby programming", "programming_languages"),
    ("swift", "swift language|swift programming", "programming_languages"),
    ("kotlin", "kotlin programming", "programming_languages"),
    ("scala", "scala programming", "programming_languages"),
    ("perl", "perl programming", "programming_languages"),
    ("bash", "bash scripting|bash shell|bourne-again shell", "programming_languages"),
    ("shell", "shell scripting|sh", "programming_languages"),
    ("powershell", "posh|ps1", "programming_languages"),
    ("dart", "dart language|dart programming", "programming_languages"),
    ("lua", "lua programming", "programming_languages"),
    ("haskell", "haskell programming", "programming_languages"),
    ("clojure", "clojure programming", "programming_languages"),
    ("elixir", "elixir programming", "programming_languages"),
    ("erlang", "erlang otp", "programming_languages"),
    ("julia", "julia language|julia programming", "programming_languages"),
    ("matlab", "matlab programming", "programming_languages"),
    ("vba", "visual basic for applications|excel vba", "programming_languages"),
    ("visual basic", "vb|vb.net", "programming_languages"),
    ("objective-c", "obj-c|objective c", "programming_languages"),
    ("groovy", "apache groovy", "programming_languages"),
    ("fortran", "fortran 90|fortran 77", "programming_languages"),
    ("cobol", "cobol programming", "programming_languages"),
    ("assembly", "asm|assembly language|x86 assembly", "programming_languages"),
    ("sql", "structured query language|ansi sql", "programming_languages"),

    # 2. Web & Frameworks
    ("react", "react.js|reactjs", "web_frameworks"),
    ("angular", "angular.js|angularjs|angular2+", "web_frameworks"),
    ("vue.js", "vue|vuejs|vue 3", "web_frameworks"),
    ("node.js", "nodejs|node", "web_frameworks"),
    ("django", "django framework", "web_frameworks"),
    ("flask", "flask framework", "web_frameworks"),
    ("fastapi", "fastapi framework", "web_frameworks"),
    ("express.js", "express|expressjs", "web_frameworks"),
    ("spring boot", "spring-boot", "web_frameworks"),
    ("spring", "spring framework", "web_frameworks"),
    ("asp.net", "asp.net core|aspnet", "web_frameworks"),
    ("svelte", "sveltejs|sveltekit", "web_frameworks"),
    ("next.js", "nextjs|next", "web_frameworks"),
    ("nuxt.js", "nuxtjs|nuxt", "web_frameworks"),
    ("ruby on rails", "rails|ror", "web_frameworks"),
    ("laravel", "laravel framework", "web_frameworks"),
    ("symfony", "symfony framework", "web_frameworks"),
    ("bootstrap", "twitter bootstrap|bootstrap 5", "web_frameworks"),
    ("tailwind css", "tailwind|tailwindcss", "web_frameworks"),
    ("html5", "html", "web_frameworks"),
    ("css3", "css", "web_frameworks"),
    ("sass", "scss", "web_frameworks"),
    ("graphql", "gql", "web_frameworks"),
    ("rest", "rest api|restful|restful apis", "web_frameworks"),
    ("grpc", "grpc framework|rpc", "web_frameworks"),
    ("redux", "redux toolkit|rtk", "web_frameworks"),
    ("jquery", "jquery library", "web_frameworks"),
    ("electron", "electron.js|electronjs", "web_frameworks"),
    ("flutter", "flutter framework|flutter sdk", "web_frameworks"),
    ("react native", "react-native|rn", "web_frameworks"),
    ("websocket", "websockets|ws", "web_frameworks"),
    ("webpack", "webpack bundler", "web_frameworks"),
    ("vite", "vitejs", "web_frameworks"),
    ("fastify", "fastify framework", "web_frameworks"),

    # 3. Data & ML/DL
    ("machine learning", "ml", "data_ml_dl"),
    ("deep learning", "dl", "data_ml_dl"),
    ("artificial intelligence", "ai", "data_ml_dl"),
    ("natural language processing", "nlp", "data_ml_dl"),
    ("computer vision", "cv", "data_ml_dl"),
    ("reinforcement learning", "rl", "data_ml_dl"),
    ("pytorch", "torch", "data_ml_dl"),
    ("tensorflow", "tf", "data_ml_dl"),
    ("keras", "keras framework", "data_ml_dl"),
    ("scikit-learn", "sklearn", "data_ml_dl"),
    ("pandas", "pandas library", "data_ml_dl"),
    ("numpy", "numpy library", "data_ml_dl"),
    ("scipy", "scipy library", "data_ml_dl"),
    ("spark", "apache spark|pyspark", "data_ml_dl"),
    ("hadoop", "apache hadoop|hdfs", "data_ml_dl"),
    ("kafka", "apache kafka", "data_ml_dl"),
    ("flink", "apache flink", "data_ml_dl"),
    ("dbt", "data build tool", "data_ml_dl"),
    ("airflow", "apache airflow", "data_ml_dl"),
    ("mlflow", "mlflow tracking", "data_ml_dl"),
    ("hugging face", "huggingface|transformers", "data_ml_dl"),
    ("large language models", "llm|llms", "data_ml_dl"),
    ("langchain", "langchain framework", "data_ml_dl"),
    ("generative ai", "genai", "data_ml_dl"),
    ("data modeling", "dimensional modeling", "data_ml_dl"),
    ("data warehousing", "dwh|edw", "data_ml_dl"),
    ("etl", "extract transform load|elt", "data_ml_dl"),
    ("data pipelines", "data pipeline", "data_ml_dl"),
    ("data visualization", "dataviz", "data_ml_dl"),
    ("opencv", "opencv library", "data_ml_dl"),
    ("spacy", "spacy nlp", "data_ml_dl"),
    ("nltk", "natural language toolkit", "data_ml_dl"),
    ("xgboost", "xgb", "data_ml_dl"),
    ("lightgbm", "lgbm", "data_ml_dl"),
    ("statsmodels", "statsmodels library", "data_ml_dl"),
    ("time series analysis", "time series forecasting|forecasting", "data_ml_dl"),
    ("feature engineering", "feature selection", "data_ml_dl"),
    ("ab testing", "a/b testing|split testing|experimentation", "data_ml_dl"),
    ("statistical analysis", "applied statistics|inferential statistics", "data_ml_dl"),

    # 4. Databases
    ("postgresql", "postgres|psql", "databases"),
    ("mysql", "my sql", "databases"),
    ("sqlite", "sqlite3", "databases"),
    ("microsoft sql server", "ms sql|sql server|mssql|t-sql", "databases"),
    ("oracle database", "oracle db|pl/sql", "databases"),
    ("mongodb", "mongo", "databases"),
    ("redis", "redis cache", "databases"),
    ("elasticsearch", "es|elastic search", "databases"),
    ("cassandra", "apache cassandra", "databases"),
    ("dynamodb", "amazon dynamodb", "databases"),
    ("snowflake", "snowflake dwh|snowflake data cloud", "databases"),
    ("bigquery", "google bigquery|bq", "databases"),
    ("redshift", "amazon redshift", "databases"),
    ("neo4j", "neo4j graph", "databases"),
    ("couchdb", "apache couchdb", "databases"),
    ("mariadb", "maria db", "databases"),
    ("clickhouse", "clickhouse db", "databases"),
    ("databricks", "databricks lakehouse", "databases"),
    ("hbase", "apache hbase", "databases"),
    ("access", "ms access|microsoft access", "databases"),
    ("firestore", "google cloud firestore", "databases"),
    ("supabase", "supabase db", "databases"),
    ("vector databases", "vector search|pinecone|weaviate|chromadb|milvus", "databases"),

    # 5. Cloud & DevOps
    ("amazon web services", "aws", "cloud_devops"),
    ("microsoft azure", "azure", "cloud_devops"),
    ("google cloud platform", "gcp|google cloud", "cloud_devops"),
    ("docker", "docker container|docker containers", "cloud_devops"),
    ("kubernetes", "k8s", "cloud_devops"),
    ("terraform", "hashicorp terraform", "cloud_devops"),
    ("ansible", "ansible automation", "cloud_devops"),
    ("jenkins", "jenkins ci", "cloud_devops"),
    ("github actions", "gh actions", "cloud_devops"),
    ("gitlab ci", "gitlab ci/cd", "cloud_devops"),
    ("ci/cd", "continuous integration|continuous deployment", "cloud_devops"),
    ("linux", "gnu/linux|ubuntu|centos|rhel|debian", "cloud_devops"),
    ("unix", "unix system", "cloud_devops"),
    ("helm", "helm charts", "cloud_devops"),
    ("prometheus", "prometheus monitoring", "cloud_devops"),
    ("grafana", "grafana dashboards", "cloud_devops"),
    ("datadog", "datadog monitoring", "cloud_devops"),
    ("splunk", "splunk enterprise", "cloud_devops"),
    ("cloudformation", "aws cloudformation", "cloud_devops"),
    ("lambda", "aws lambda|serverless", "cloud_devops"),
    ("openstack", "openstack cloud", "cloud_devops"),
    ("vmware", "vmware vsphere|esxi", "cloud_devops"),
    ("vagrant", "hashicorp vagrant", "cloud_devops"),
    ("nginx", "nginx server", "cloud_devops"),
    ("apache http server", "apache web server|httpd", "cloud_devops"),
    ("circleci", "circle ci", "cloud_devops"),
    ("argo cd", "argocd", "cloud_devops"),
    ("sonarqube", "sonar", "cloud_devops"),
    ("snyk", "snyk security", "cloud_devops"),
    ("microservices", "microservice architecture", "cloud_devops"),
    ("site reliability engineering", "sre", "cloud_devops"),
    ("infrastructure as code", "iac", "cloud_devops"),
    ("system administration", "sysadmin", "cloud_devops"),
    ("devsecops", "dev sec ops", "cloud_devops"),
    ("puppet", "puppet software|puppet devops", "cloud_devops"),
    ("chef", "chef software|chef devops", "cloud_devops"),

    # 6. Tools
    ("git", "git vcs", "software_tools"),
    ("github", "gh", "software_tools"),
    ("gitlab", "gitlab vcs", "software_tools"),
    ("bitbucket", "atlassian bitbucket", "software_tools"),
    ("jira", "atlassian jira", "software_tools"),
    ("confluence", "atlassian confluence", "software_tools"),
    ("figma", "figma design", "software_tools"),
    ("postman", "postman api", "software_tools"),
    ("visual studio code", "vs code|vscode", "software_tools"),
    ("visual studio", "vs ide", "software_tools"),
    ("pycharm", "jetbrains pycharm", "software_tools"),
    ("intellij idea", "intellij", "software_tools"),
    ("eclipse", "eclipse ide", "software_tools"),
    ("jupyter", "jupyter notebook|jupyterlab", "software_tools"),
    ("excel", "ms excel|microsoft excel|spreadsheets", "software_tools"),
    ("word", "ms word|microsoft word", "software_tools"),
    ("powerpoint", "ms powerpoint|microsoft powerpoint|ppt", "software_tools"),
    ("outlook", "ms outlook|microsoft outlook", "software_tools"),
    ("power bi", "powerbi|pbi", "software_tools"),
    ("tableau", "tableau software|tableau desktop", "software_tools"),
    ("looker", "google looker", "software_tools"),
    ("trello", "trello boards", "software_tools"),
    ("slack", "slack messaging", "software_tools"),
    ("asana", "asana project", "software_tools"),
    ("monday.com", "monday", "software_tools"),
    ("notion", "notion workspace", "software_tools"),
    ("adobe photoshop", "photoshop", "software_tools"),
    ("adobe illustrator", "illustrator", "software_tools"),
    ("adobe xd", "xd", "software_tools"),
    ("blender", "blender 3d", "software_tools"),
    ("make", "gnu make|makefile", "software_tools"),
    ("yarn", "yarn package manager", "software_tools"),

    # 7. Accounting & Finance
    ("financial reporting", "financial statements", "accounting_finance"),
    ("general ledger", "gl accounting", "accounting_finance"),
    ("balance sheet", "balance sheet preparation", "accounting_finance"),
    ("income statement", "profit and loss|p&l", "accounting_finance"),
    ("cash flow statement", "cash flow forecasting|cash flow management", "accounting_finance"),
    ("bank reconciliation", "account reconciliation", "accounting_finance"),
    ("accounts payable", "ap accounting", "accounting_finance"),
    ("accounts receivable", "ar accounting", "accounting_finance"),
    ("journal entries", "adjusting entries", "accounting_finance"),
    ("chart of accounts", "coa", "accounting_finance"),
    ("financial auditing", "internal audit|external audit", "accounting_finance"),
    ("gaap", "us gaap|generally accepted accounting principles", "accounting_finance"),
    ("ifrs", "international financial reporting standards", "accounting_finance"),
    ("sox compliance", "sarbanes-oxley|sox", "accounting_finance"),
    ("tax preparation", "tax returns|tax filing", "accounting_finance"),
    ("corporate tax", "business tax", "accounting_finance"),
    ("payroll processing", "payroll management|payroll systems", "accounting_finance"),
    ("quickbooks", "intuit quickbooks|qbo", "accounting_finance"),
    ("xero", "xero accounting", "accounting_finance"),
    ("financial modeling", "dcf modeling|lbo modeling", "accounting_finance"),
    ("financial analysis", "fp&a|financial planning and analysis", "accounting_finance"),
    ("variance analysis", "budget vs actual", "accounting_finance"),
    ("cost accounting", "cost analysis|job costing", "accounting_finance"),
    ("capital budgeting", "capex|capital expenditures", "accounting_finance"),
    ("working capital management", "working capital", "accounting_finance"),
    ("fixed asset accounting", "depreciation schedules", "accounting_finance"),
    ("equity research", "stock analysis", "accounting_finance"),
    ("portfolio management", "asset allocation", "accounting_finance"),
    ("risk management", "credit risk|market risk|operational risk", "accounting_finance"),
    ("credit analysis", "credit underwriting", "accounting_finance"),
    ("valuation", "business valuation|company valuation", "accounting_finance"),
    ("mergers and acquisitions", "m&a", "accounting_finance"),
    ("investment banking", "corporate finance", "accounting_finance"),
    ("treasury management", "liquidity management", "accounting_finance"),
    ("sap erp", "sap fico", "accounting_finance"),
    ("oracle netsuite", "netsuite", "accounting_finance"),
    ("cpa", "certified public accountant", "accounting_finance"),
    ("cfa", "chartered financial analyst", "accounting_finance"),
    ("cost reduction", "expense management", "accounting_finance"),
    ("forensic accounting", "fraud examination", "accounting_finance"),

    # 8. Healthcare & Medical
    ("patient care", "direct patient care|bedside care", "healthcare_medical"),
    ("vital signs", "vital signs monitoring|vitals", "healthcare_medical"),
    ("cardiopulmonary resuscitation", "cpr", "healthcare_medical"),
    ("basic life support", "bls", "healthcare_medical"),
    ("advanced cardiac life support", "acls", "healthcare_medical"),
    ("electronic health records", "ehr", "healthcare_medical"),
    ("electronic medical records", "emr", "healthcare_medical"),
    ("epic systems", "epic ehr|epic software", "healthcare_medical"),
    ("cerner", "cerner ehr|cerner software", "healthcare_medical"),
    ("triage", "patient triage|emergency triage", "healthcare_medical"),
    ("phlebotomy", "venipuncture|blood draw", "healthcare_medical"),
    ("pharmacology", "drug dosage calculations|pharmacokinetics", "healthcare_medical"),
    ("medication administration", "med pass|medication dispensing", "healthcare_medical"),
    ("intravenous therapy", "iv therapy|iv insertion", "healthcare_medical"),
    ("patient assessment", "clinical assessment|physical assessment", "healthcare_medical"),
    ("infection control", "sterile technique|aseptic technique", "healthcare_medical"),
    ("wound care", "dressing changes|wound dressing", "healthcare_medical"),
    ("catheterization", "urinary catheterization|foley catheter", "healthcare_medical"),
    ("hipaa compliance", "hipaa regulations|patient privacy", "healthcare_medical"),
    ("medical terminology", "medical coding", "healthcare_medical"),
    ("icd-10", "icd-10-cm|icd coding", "healthcare_medical"),
    ("cpt coding", "current procedural terminology", "healthcare_medical"),
    ("medical billing", "claims processing", "healthcare_medical"),
    ("patient education", "discharge planning", "healthcare_medical"),
    ("telemetry", "telemetry monitoring|ecg interpretation|ekg", "healthcare_medical"),
    ("clinical documentation", "charting|progress notes", "healthcare_medical"),
    ("palliative care", "hospice care", "healthcare_medical"),
    ("pediatric care", "pediatrics", "healthcare_medical"),
    ("geriatric care", "gerontology", "healthcare_medical"),
    ("surgical preparation", "pre-op care|post-op care|pacu", "healthcare_medical"),
    ("dialysis", "hemodialysis|peritoneal dialysis", "healthcare_medical"),
    ("radiology", "x-ray imaging|ct scan|mri", "healthcare_medical"),
    ("sonography", "ultrasound", "healthcare_medical"),
    ("rehabilitation", "physical rehab", "healthcare_medical"),
    ("behavioral health", "psychiatric care|mental health assessment", "healthcare_medical"),
    ("patient advocacy", "care coordination", "healthcare_medical"),
    ("acute care", "critical care|icu care", "healthcare_medical"),
    ("first aid", "emergency first aid", "healthcare_medical"),
    ("clinical research", "clinical trials|gcp guidelines", "healthcare_medical"),
    ("chemotherapy administration", "oncology nursing", "healthcare_medical"),

    # 9. Teaching & Education
    ("curriculum development", "curriculum design|curriculum planning", "teaching_education"),
    ("lesson planning", "lesson plan design", "teaching_education"),
    ("classroom management", "behavior management", "teaching_education"),
    ("student assessment", "formative assessment|summative assessment", "teaching_education"),
    ("differentiated instruction", "individualized learning", "teaching_education"),
    ("instructional design", "addie model", "teaching_education"),
    ("special education", "sped|individualized education plan|iep", "teaching_education"),
    ("educational technology", "edtech|learning management systems|lms", "teaching_education"),
    ("canvas lms", "canvas", "teaching_education"),
    ("blackboard", "blackboard learn", "teaching_education"),
    ("google classroom", "google workspace for education", "teaching_education"),
    ("pedagogical methods", "pedagogy|teaching methodologies", "teaching_education"),
    ("literacy instruction", "reading instruction|phonics", "teaching_education"),
    ("stem education", "stem curriculum", "teaching_education"),
    ("bilingual education", "esl|esol|ell instruction", "teaching_education"),
    ("standardized testing", "test preparation|exam proctoring", "teaching_education"),
    ("grading", "rubric development|grading papers", "teaching_education"),
    ("parent communication", "parent-teacher conferences", "teaching_education"),
    ("student engagement", "active learning strategies", "teaching_education"),
    ("tutoring", "one-on-one tutoring|academic tutoring", "teaching_education"),
    ("early childhood education", "preschool teaching", "teaching_education"),
    ("higher education instruction", "college teaching|university lecturing", "teaching_education"),
    ("syllabus design", "course syllabus", "teaching_education"),
    ("student advising", "academic advising", "teaching_education"),
    ("educational leadership", "school administration", "teaching_education"),
    ("distance learning", "remote teaching|online instruction", "teaching_education"),
    ("social emotional learning", "sel", "teaching_education"),
    ("inquiry-based learning", "project-based learning|pbl", "teaching_education"),
    ("peer mentoring", "student mentorship", "teaching_education"),
    ("academic research", "scholarly writing", "teaching_education"),

    # 10. Sales & Marketing
    ("lead generation", "prospecting|cold outreach", "sales_marketing"),
    ("cold calling", "telemarketing", "sales_marketing"),
    ("b2b sales", "business-to-business sales", "sales_marketing"),
    ("b2c sales", "business-to-consumer sales", "sales_marketing"),
    ("salesforce", "salesforce crm|sfdc", "sales_marketing"),
    ("hubspot", "hubspot crm", "sales_marketing"),
    ("sales pipeline management", "pipeline management|sales funnel", "sales_marketing"),
    ("contract negotiation", "deal closing|closing techniques", "sales_marketing"),
    ("account management", "key account management|client retention", "sales_marketing"),
    ("consultative selling", "solution selling", "sales_marketing"),
    ("consultative sales", "consultative approach", "sales_marketing"),
    ("customer relationship management", "crm systems", "sales_marketing"),
    ("sales forecasting", "revenue forecasting", "sales_marketing"),
    ("quota attainment", "sales targets", "sales_marketing"),
    ("territory management", "sales territory planning", "sales_marketing"),
    ("business development", "bizdev", "sales_marketing"),
    ("inside sales", "outbound sales", "sales_marketing"),
    ("retail sales", "merchandise sales|floor sales", "sales_marketing"),
    ("point of sale", "pos systems", "sales_marketing"),
    ("search engine optimization", "seo", "sales_marketing"),
    ("search engine marketing", "sem|pay-per-click|ppc", "sales_marketing"),
    ("google analytics", "ga4|universal analytics", "sales_marketing"),
    ("google ads", "google adwords", "sales_marketing"),
    ("social media marketing", "smm", "sales_marketing"),
    ("content marketing", "content strategy", "sales_marketing"),
    ("email marketing", "newsletter marketing|mailchimp", "sales_marketing"),
    ("marketing automation", "marketo", "sales_marketing"),
    ("brand management", "brand awareness|brand positioning", "sales_marketing"),
    ("market research", "competitive analysis|market intelligence", "sales_marketing"),
    ("customer journey mapping", "conversion rate optimization|cro", "sales_marketing"),
    ("public relations", "pr campaigns|press releases", "sales_marketing"),
    ("copywriting", "ad copywriting|promotional writing", "sales_marketing"),
    ("digital marketing", "online marketing", "sales_marketing"),
    ("sales enablement", "sales collateral", "sales_marketing"),
    ("merchandising", "visual merchandising|product display", "sales_marketing"),
    ("upselling", "cross-selling", "sales_marketing"),
    ("sales", "selling|direct sales|sales operations", "sales_marketing"),

    # 11. HR & Recruiting
    ("talent acquisition", "recruitment|recruiting", "hr_recruiting"),
    ("candidate sourcing", "headhunting|boolean search", "hr_recruiting"),
    ("applicant tracking systems", "ats platforms", "hr_recruiting"),
    ("workday hcm", "workday hr", "hr_recruiting"),
    ("employee onboarding", "new hire orientation", "hr_recruiting"),
    ("employee relations", "workplace conflict resolution", "hr_recruiting"),
    ("performance management", "performance appraisals|annual reviews", "hr_recruiting"),
    ("compensation and benefits", "comp & benefits|total rewards", "hr_recruiting"),
    ("job descriptions", "job posting creation", "hr_recruiting"),
    ("structured interviewing", "behavioral interviewing", "hr_recruiting"),
    ("background checks", "employment verification", "hr_recruiting"),
    ("hr policies", "employee handbook development", "hr_recruiting"),
    ("exit interviews", "offboarding process", "hr_recruiting"),
    ("succession planning", "talent management", "hr_recruiting"),
    ("diversity equity inclusion", "dei initiatives|diversity hiring", "hr_recruiting"),
    ("employee engagement", "pulse surveys|retention strategies", "hr_recruiting"),
    ("labor law compliance", "eeoc|fmla|flsa", "hr_recruiting"),
    ("human resources information systems", "hris platforms", "hr_recruiting"),
    ("payroll administration", "benefits enrollment", "hr_recruiting"),
    ("union relations", "collective bargaining|labor relations", "hr_recruiting"),
    ("training and development", "learning and development|l&d", "hr_recruiting"),
    ("workforce planning", "headcount planning", "hr_recruiting"),
    ("hr analytics", "people analytics|workforce analytics", "hr_recruiting"),
    ("employer branding", "recruitment marketing", "hr_recruiting"),

    # 12. Culinary & Chef
    ("culinary arts", "cooking techniques", "culinary_chef"),
    ("food preparation", "prep work|mise en place", "culinary_chef"),
    ("knife skills", "chopping|julienne|dicing", "culinary_chef"),
    ("food safety", "safe food handling|servsafe", "culinary_chef"),
    ("haccp", "hazard analysis critical control points", "culinary_chef"),
    ("menu planning", "menu engineering|menu design", "culinary_chef"),
    ("recipe development", "standardized recipes", "culinary_chef"),
    ("kitchen sanitation", "commercial kitchen cleaning", "culinary_chef"),
    ("food plating", "food presentation|garnishing", "culinary_chef"),
    ("inventory control", "kitchen inventory|food cost control", "culinary_chef"),
    ("baking", "bread making|artisan breads", "culinary_chef"),
    ("pastry arts", "patisserie|dessert preparation", "culinary_chef"),
    ("butchery", "meat fabrication|poultry fabrication", "culinary_chef"),
    ("seafood preparation", "fish filleting", "culinary_chef"),
    ("sous vide cooking", "precision cooking", "culinary_chef"),
    ("sautéing", "pan frying", "culinary_chef"),
    ("grilling", "charbroiling|bbq", "culinary_chef"),
    ("roasting", "slow roasting", "culinary_chef"),
    ("sauce making", "mother sauces|reductions", "culinary_chef"),
    ("catering", "banquet service|large event cooking", "culinary_chef"),
    ("kitchen equipment operation", "commercial ovens|fryers|slicers", "culinary_chef"),
    ("food waste reduction", "portion control", "culinary_chef"),
    ("expediting", "line expeditor", "culinary_chef"),
    ("station management", "line cook stations", "culinary_chef"),
    ("culinary nutrition", "special diets|allergen management", "culinary_chef"),

    # 13. Fitness & Wellness
    ("personal training", "fitness coaching", "fitness_wellness"),
    ("strength training", "weightlifting|resistance training", "fitness_wellness"),
    ("cardiovascular conditioning", "cardio workouts|aerobics", "fitness_wellness"),
    ("functional fitness", "functional training", "fitness_wellness"),
    ("nutrition counseling", "dietary planning|macronutrient tracking", "fitness_wellness"),
    ("body composition assessment", "skinfold measurements|bioimpedance", "fitness_wellness"),
    ("exercise physiology", "biomechanics", "fitness_wellness"),
    ("injury rehabilitation", "corrective exercise", "fitness_wellness"),
    ("group fitness instruction", "group exercise", "fitness_wellness"),
    ("high intensity interval training", "hiit", "fitness_wellness"),
    ("flexibility training", "stretching protocols|mobility drills", "fitness_wellness"),
    ("yoga instruction", "hatha yoga|vinyasa yoga", "fitness_wellness"),
    ("pilates instruction", "mat pilates|reformer pilates", "fitness_wellness"),
    ("athletic conditioning", "sports performance training", "fitness_wellness"),
    ("client fitness assessment", "physical fitness testing", "fitness_wellness"),
    ("workout program design", "exercise prescription", "fitness_wellness"),
    ("wellness coaching", "lifestyle modification", "fitness_wellness"),
    ("kettlebell training", "kettlebell exercises", "fitness_wellness"),
    ("core conditioning", "core strengthening", "fitness_wellness"),
    ("weight management", "fat loss guidance", "fitness_wellness"),
    ("cpr certification", "cpr/aed", "fitness_wellness"),
    ("sports nutrition", "athletic fueling", "fitness_wellness"),

    # 14. Legal & Compliance
    ("legal research", "statutory research|case law research", "legal_compliance"),
    ("legal writing", "brief writing|memorandum drafting", "legal_compliance"),
    ("litigation support", "trial preparation", "legal_compliance"),
    ("contract drafting", "contract review|agreement drafting", "legal_compliance"),
    ("due diligence", "corporate due diligence", "legal_compliance"),
    ("westlaw", "westlaw edge", "legal_compliance"),
    ("lexisnexis", "lexis", "legal_compliance"),
    ("deposition preparation", "witness preparation", "legal_compliance"),
    ("discovery process", "e-discovery|document review", "legal_compliance"),
    ("regulatory compliance", "regulatory filings", "legal_compliance"),
    ("intellectual property", "patent filing|trademark registration", "legal_compliance"),
    ("corporate governance", "board resolutions|bylaws", "legal_compliance"),
    ("case management", "legal case tracking", "legal_compliance"),
    ("client counseling", "legal advisory", "legal_compliance"),
    ("ethics compliance", "professional responsibility", "legal_compliance"),
    ("anti-money laundering", "aml compliance|kyc", "legal_compliance"),
    ("data privacy compliance", "gdpr|ccpa", "legal_compliance"),
    ("pleadings", "motions|legal pleadings", "legal_compliance"),
    ("paralegal support", "legal assistant tasks", "legal_compliance"),
    ("trademark law", "copyright registration", "legal_compliance"),

    # 15. Engineering & Construction
    ("blueprint reading", "architectural drawings|schematic reading", "engineering_construction"),
    ("computer-aided design", "cad drafting", "engineering_construction"),
    ("autocad", "autodesk autocad", "engineering_construction"),
    ("solidworks", "dassault solidworks", "engineering_construction"),
    ("revit", "autodesk revit|bim", "engineering_construction"),
    ("civil engineering", "civil infrastructure", "engineering_construction"),
    ("mechanical engineering", "mechanical systems design", "engineering_construction"),
    ("electrical engineering", "circuit design|schematics", "engineering_construction"),
    ("structural analysis", "finite element analysis|fea", "engineering_construction"),
    ("cost estimating", "construction takeoff|bid preparation", "engineering_construction"),
    ("project scheduling", "construction scheduling", "engineering_construction"),
    ("quality assurance", "qa/qc inspection", "engineering_construction"),
    ("osha compliance", "construction safety|site safety", "engineering_construction"),
    ("hvac systems", "heating ventilation air conditioning", "engineering_construction"),
    ("plumbing systems", "pipefitting", "engineering_construction"),
    ("electrical wiring", "panel wiring|conduit bending", "engineering_construction"),
    ("welding", "mig welding|tig welding|arc welding", "engineering_construction"),
    ("carpentry", "framing|finish carpentry", "engineering_construction"),
    ("concrete work", "concrete pouring|rebar placement", "engineering_construction"),
    ("heavy machinery operation", "excavator|forklift|crane", "engineering_construction"),
    ("surveying", "total station|gps surveying", "engineering_construction"),
    ("building codes", "ibc compliance|zoning regulations", "engineering_construction"),
    ("commissioning", "facility commissioning", "engineering_construction"),
    ("materials testing", "soil compaction|concrete testing", "engineering_construction"),
    ("lean construction", "last planner system", "engineering_construction"),

    # 16. Soft & Professional Skills
    ("communication", "verbal communication|written communication", "soft_skills"),
    ("leadership", "team leadership|people leadership", "soft_skills"),
    ("problem solving", "troubleshooting|analytical problem solving", "soft_skills"),
    ("critical thinking", "logical reasoning", "soft_skills"),
    ("teamwork", "collaboration|cross-functional teamwork", "soft_skills"),
    ("time management", "task prioritization", "soft_skills"),
    ("adaptability", "flexibility|change management", "soft_skills"),
    ("emotional intelligence", "eq|empathy", "soft_skills"),
    ("conflict resolution", "mediation|conflict management", "soft_skills"),
    ("negotiation", "persuasion|influencing skills", "soft_skills"),
    ("presentation skills", "public speaking|oral presentation", "soft_skills"),
    ("active listening", "empathic listening", "soft_skills"),
    ("attention to detail", "accuracy|thoroughness", "soft_skills"),
    ("organizational skills", "multitasking|organization", "soft_skills"),
    ("decision making", "strategic decision making", "soft_skills"),
    ("customer service", "client satisfaction|customer relations", "soft_skills"),
    ("project management", "pmp|agile project management", "soft_skills"),
    ("scrum", "scrum master|sprint planning", "soft_skills"),
    ("kanban", "kanban board", "soft_skills"),
    ("stakeholder management", "stakeholder engagement", "soft_skills"),
    ("strategic planning", "business strategy|okrs", "soft_skills"),
    ("mentoring", "coaching|mentorship", "soft_skills"),
    ("work ethic", "professionalism|dependability", "soft_skills"),
    ("project", "project coordination|project work", "soft_skills"),
    ("management", "general management|team management", "soft_skills"),
]


# -- Ambiguous Terms (Terms colliding with ordinary English words) ----------
AMBIGUOUS_TERMS: List[Tuple[str, str]] = [
    ("c", "require_context"),
    ("r", "require_context"),
    ("go", "require_context"),
    ("java", "case_sensitive"),
    ("swift", "require_context"),
    ("rust", "require_context"),
    ("spark", "require_context"),
    ("excel", "require_context"),
    ("word", "require_context"),
    ("access", "require_context"),
    ("sales", "exclude_standalone"),
    ("project", "exclude_standalone"),
    ("management", "exclude_standalone"),
    ("chef", "require_context"),
    ("puppet", "require_context"),
    ("spring", "require_context"),
    ("react", "require_context"),
    ("make", "exclude_standalone"),
    ("bash", "require_context"),
    ("git", "require_context"),
    ("rest", "require_context"),
    ("dart", "require_context"),
    ("sass", "require_context"),
    ("yarn", "require_context"),
    ("jira", "case_sensitive"),
    ("triage", "require_context"),
    ("welding", "require_context"),
    ("baking", "require_context"),
    ("grilling", "require_context"),
    ("roasting", "require_context"),
    ("communication", "exclude_standalone"),
    ("leadership", "exclude_standalone"),
    ("teamwork", "exclude_standalone"),
]


# -- O*NET Element to Category Mapping --------------------------------------

def map_onet_element_to_category(elem: str) -> str:
    """Map an O*NET Element Name to a unified category string.

    Args:
        elem: O*NET Element Name string.

    Returns:
        Unified category name.
    """
    elem = (elem or "").lower()
    if any(k in elem for k in ["data base", "database", "data mining", "metadata", "query"]):
        return "databases"
    if any(k in elem for k in ["analytical", "scientific", "business intelligence", "analysis software"]):
        return "data_ml_dl"
    if any(k in elem for k in ["cad", "cam", "geographic", "map creation"]):
        return "cad_engineering"
    if any(k in elem for k in ["medical", "health"]):
        return "healthcare_medical"
    if any(k in elem for k in ["accounting", "financial", "tax preparation", "billing"]):
        return "accounting_finance"
    if any(k in elem for k in ["human resources", "hr"]):
        return "hr_recruiting"
    if any(k in elem for k in ["customer relationship", "crm", "sales", "point of sale", "pos"]):
        return "sales_marketing"
    if any(k in elem for k in ["project management", "time accounting", "calendar"]):
        return "project_management"
    if any(k in elem for k in ["graphics", "photo", "imaging", "video", "sound", "music", "desktop publishing"]):
        return "design_multimedia"
    if any(k in elem for k in ["development environment", "object or component", "compiler", "program testing"]):
        return "programming_languages"
    if any(k in elem for k in ["web platform", "web page"]):
        return "web_frameworks"
    if any(k in elem for k in ["cloud", "operating system", "configuration management", "file versioning", "clustering"]):
        return "cloud_devops"
    if any(k in elem for k in ["network", "switch", "router", "connectivity"]):
        return "networking"
    if any(k in elem for k in ["security", "protection", "authentication", "virus"]):
        return "cybersecurity"
    if any(k in elem for k in ["erp", "enterprise", "procurement", "supply chain", "logistics", "inventory"]):
        return "enterprise_software"
    if any(k in elem for k in ["office suite", "word processing", "spreadsheet", "presentation", "electronic mail", "browser", "document management", "instant messaging"]):
        return "software_tools"
    return "software_tools"


# -- Synthetic NER Industry to Category Mapping -----------------------------

def map_synth_industry_to_category(industry: str) -> str:
    """Map a synthetic NER industry to a unified category string.

    Args:
        industry: Synthetic resume industry string.

    Returns:
        Unified category name.
    """
    ind = (industry or "").lower()
    mapping = {
        "software": "programming_languages",
        "data": "data_ml_dl",
        "cybersecurity": "cybersecurity",
        "telecommunications": "networking",
        "logistics": "operations_logistics",
        "transportation": "operations_logistics",
        "healthcare": "healthcare_medical",
        "biotechnology": "healthcare_medical",
        "marketing": "sales_marketing",
        "retail": "sales_marketing",
        "finance": "accounting_finance",
        "legal": "legal_compliance",
        "education": "teaching_education",
        "hospitality": "culinary_chef",
        "design": "design_multimedia",
        "media": "design_multimedia",
        "construction": "engineering_construction",
        "manufacturing": "engineering_construction",
        "energy": "engineering_construction",
        "agriculture": "agriculture",
    }
    return mapping.get(ind, "other")


# -- Helpers ----------------------------------------------------------------

def clean_key(s: str) -> str:
    """Normalize skill string: lowercase and stripped."""
    return re.sub(r"\s+", " ", str(s).strip().lower())


def word_count(s: str) -> int:
    """Count whitespace-separated words."""
    return len(str(s).split())


# -- Main Pipeline ----------------------------------------------------------

def build_taxonomy() -> None:
    """Build, clean, and write the unified skills taxonomy."""
    logger.info("=== build_skills_taxonomy.py start ===")

    # 1. Write skills_curated.csv
    curated_rows: List[Dict[str, str]] = []
    seen_curated_skills: Set[str] = set()
    curated_alias_to_skill: Dict[str, str] = {}

    for skill, aliases, cat in RAW_CURATED_SKILLS:
        s_clean = clean_key(skill)
        if s_clean in seen_curated_skills:
            continue
        seen_curated_skills.add(s_clean)

        # Parse & clean aliases
        clean_aliases: List[str] = []
        if aliases:
            for a in aliases.split("|"):
                a_clean = clean_key(a)
                if a_clean and a_clean != s_clean and a_clean not in curated_alias_to_skill:
                    clean_aliases.append(a_clean)
                    curated_alias_to_skill[a_clean] = s_clean

        curated_rows.append({
            "skill": s_clean,
            "aliases": "|".join(clean_aliases),
            "category": cat,
        })

    curated_df = pd.DataFrame(curated_rows)
    curated_df.to_csv(CURATED_PATH, index=False)
    logger.info(f"Wrote {CURATED_PATH} ({len(curated_df)} curated skills)")

    # 2. Write ambiguous_terms.csv
    ambig_rows: List[Dict[str, str]] = []
    seen_ambig: Set[str] = set()
    for term, rule in AMBIGUOUS_TERMS:
        t_clean = clean_key(term)
        if t_clean not in seen_ambig:
            ambig_rows.append({"term": t_clean, "rule": rule})
            seen_ambig.add(t_clean)

    ambig_df = pd.DataFrame(ambig_rows)
    ambig_df.to_csv(AMBIGUOUS_PATH, index=False)
    logger.info(f"Wrote {AMBIGUOUS_PATH} ({len(ambig_df)} ambiguous terms)")

    # 3. Process O*NET software_skills.csv
    logger.info(f"Loading O*NET from {ONET_PATH}")
    onet = pd.read_csv(ONET_PATH)
    total_onet_entries = len(onet)
    unique_onet_examples = onet["Workplace Example"].dropna().nunique()

    occ_counts = onet.groupby("Workplace Example")["O*NET-SOC Code"].nunique()
    hot_skills = set(onet[onet["Hot Technology"] == "Y"]["Workplace Example"].dropna())

    # Filter 1: appear in >= 3 occupations or Hot Technology
    freq_or_hot = set(occ_counts[occ_counts >= 3].index) | hot_skills
    dropped_freq_or_not_hot = unique_onet_examples - len(freq_or_hot)

    # Filter 2: drop anything over 4 words
    qualifying_onet_raw = {s for s in freq_or_hot if word_count(s) <= 4}
    dropped_words_gt_4 = len(freq_or_hot) - len(qualifying_onet_raw)

    # Dominant Element Name per qualifying example
    sub_onet = onet[onet["Workplace Example"].isin(qualifying_onet_raw)]
    onet_elem_map = (
        sub_onet.groupby("Workplace Example")["Element Name"]
        .agg(lambda s: s.mode().iloc[0])
        .to_dict()
    )

    # 4. Process ner_synthetic
    logger.info(f"Loading synthetic NER skills from {SYNTH_DIR}")
    synth_skill_counts: Dict[str, Counter] = defaultdict(Counter)
    for sp in ["train", "validation", "test"]:
        p = SYNTH_DIR / f"{sp}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        for _, row in df.iterrows():
            ind = row.get("industry", "other")
            for e in row.get("entities", []):
                if e.get("label") == "SKILL":
                    synth_skill_counts[e["text"]][ind] += 1

    logger.info(f"  Distinct synthetic NER skill strings: {len(synth_skill_counts)}")

    # 5. Build Unified Taxonomy (Priority: curated -> onet -> ner_synthetic)
    # Global lookup tables to ensure:
    #   - No duplicate skill names
    #   - Every alias maps to exactly ONE skill
    #   - No alias matches any canonical skill
    canonical_skills: Dict[str, Dict[str, str]] = {}  # skill_name -> {category, aliases, source}
    alias_to_canonical: Dict[str, str] = {}           # alias -> canonical_skill

    # Helper to check if string is single character allowlist
    def is_valid_skill_string(s: str) -> bool:
        if not s:
            return False
        if len(s) == 1:
            return s in ALLOWLIST_SINGLE_CHAR
        return True

    # 5a. Insert Curated Skills
    for row in curated_rows:
        s = row["skill"]
        cat = row["category"]
        aliases_list = [a for a in row["aliases"].split("|") if a] if row["aliases"] else []
        canonical_skills[s] = {
            "aliases": aliases_list,
            "category": cat,
            "source": "curated",
        }
        for a in aliases_list:
            alias_to_canonical[a] = s

    curated_count = len(canonical_skills)
    logger.info(f"Loaded {curated_count} curated skills into taxonomy.")

    # 5b. Insert O*NET Skills
    onet_kept_count = 0
    onet_dropped_single_char = 0
    onet_merged_as_alias = 0
    onet_already_exists = 0

    for raw_ex, elem in onet_elem_map.items():
        s = clean_key(raw_ex)
        if not is_valid_skill_string(s):
            onet_dropped_single_char += 1
            continue

        cat = map_onet_element_to_category(elem)

        # Check if already exists as canonical skill
        if s in canonical_skills:
            onet_already_exists += 1
            continue

        # Check if matches an existing alias
        if s in alias_to_canonical:
            onet_merged_as_alias += 1
            continue

        # Add as new O*NET skill
        canonical_skills[s] = {
            "aliases": [],
            "category": cat,
            "source": "onet",
        }
        onet_kept_count += 1

    logger.info(f"O*NET skills kept: {onet_kept_count}")
    logger.info(f"  Already existed in curated: {onet_already_exists}")
    logger.info(f"  Merged as alias of curated: {onet_merged_as_alias}")
    logger.info(f"  Dropped single char:        {onet_dropped_single_char}")

    # 5c. Insert Synthetic NER Skills
    synth_kept_count = 0
    synth_already_exists = 0
    synth_merged_as_alias = 0

    for raw_s, counter in synth_skill_counts.items():
        s = clean_key(raw_s)
        if not is_valid_skill_string(s):
            continue

        # Top industry
        top_ind = counter.most_common(1)[0][0]
        cat = map_synth_industry_to_category(top_ind)

        if s in canonical_skills:
            synth_already_exists += 1
            continue

        if s in alias_to_canonical:
            synth_merged_as_alias += 1
            continue

        canonical_skills[s] = {
            "aliases": [],
            "category": cat,
            "source": "ner_synthetic",
        }
        synth_kept_count += 1

    logger.info(f"Synthetic NER skills kept: {synth_kept_count}")
    logger.info(f"  Already existed in curated/onet: {synth_already_exists}")
    logger.info(f"  Merged as alias of curated/onet: {synth_merged_as_alias}")

    # 6. Build DataFrame and Validate
    final_rows: List[Dict[str, str]] = []
    all_skill_names: Set[str] = set(canonical_skills.keys())

    # Final alias integrity check
    global_aliases: Set[str] = set()

    for s, data in sorted(canonical_skills.items()):
        # Filter aliases to remove collisions with primary skill names or duplicate aliases
        filtered_aliases: List[str] = []
        for a in data["aliases"]:
            if a in all_skill_names:
                continue
            if a in global_aliases:
                continue
            filtered_aliases.append(a)
            global_aliases.add(a)

        final_rows.append({
            "skill": s,
            "aliases": "|".join(filtered_aliases),
            "category": data["category"],
            "source": data["source"],
        })

    skills_df = pd.DataFrame(final_rows)
    skills_df.to_csv(SKILLS_PATH, index=False)
    logger.info(f"Wrote {SKILLS_PATH} ({len(skills_df)} total skills)")

    # 7. Print Comprehensive Report
    print("\n" + "=" * 75)
    print("SKILLS TAXONOMY REPORT")
    print("=" * 75)

    print(f"\n1. TOTAL CLEANED SKILLS: {len(skills_df)}")

    print("\n2. COUNT PER SOURCE:")
    src_counts = skills_df["source"].value_counts()
    for src, cnt in src_counts.items():
        print(f"  {src:<20} {cnt:5d}")

    print("\n3. COUNT PER CATEGORY:")
    cat_counts = skills_df["category"].value_counts()
    for cat, cnt in cat_counts.items():
        print(f"  {cat:<28} {cnt:5d}")

    print("\n4. O*NET PROCESSING BREAKDOWN:")
    print(f"  Total raw entries in software_skills.csv:     {total_onet_entries}")
    print(f"  Unique raw Workplace Examples:                 {unique_onet_examples}")
    print(f"  Dropped: <3 occupations AND not Hot Tech:      {dropped_freq_or_not_hot}")
    print(f"  Dropped: >4 words length filter:               {dropped_words_gt_4}")
    print(f"  Qualifying O*NET candidates:                   {len(qualifying_onet_raw)}")
    print(f"  Kept as primary skills in taxonomy:            {onet_kept_count}")
    print(f"  Merged/Subsumed by higher-priority curated:    {onet_already_exists + onet_merged_as_alias}")

    print("\n5. 40 RANDOM SKILLS FOR REVIEW:")
    sample = skills_df.sample(n=min(40, len(skills_df)), random_state=42)
    for i, (_, row) in enumerate(sample.iterrows(), 1):
        al = f" (aliases: {row['aliases']})" if row['aliases'] else ""
        print(f"  {i:2d}. [{row['source']:<13}] [{row['category']:<22}] {row['skill']}{al}")

    print("\n6. SKILLS COLLIDING WITH COMMON ENGLISH WORDS (Ambiguity Check):")
    colliding = []
    for s in skills_df["skill"]:
        if s in COMMON_ENGLISH_WORDS:
            colliding.append(s)
    colliding.sort()
    print(f"  Found {len(colliding)} skills matching common English words:")
    for w in colliding:
        print(f"    - {w}")

    print("\n" + "=" * 75)
    logger.info("=== build_skills_taxonomy.py done ===")


if __name__ == "__main__":
    build_taxonomy()
