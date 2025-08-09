import os
import pandas as pd
import glob
import csv

# === CONFIGURATION ===
base_dir = "PML_DATA"  # folder containing RECIPIE.csv, LINEITEMS/, SPECIFICATION/
output_dir = "PML_OUTPUT"
os.makedirs(output_dir, exist_ok=True)

# === 1. Load Recipes ===
try:
    recipe_path = os.path.join(base_dir, "RECIPIE.csv")
    recipes_df = pd.read_csv(recipe_path, sep=',', encoding='latin1', engine='python')  # changed to comma
    recipes_df.to_csv(os.path.join(output_dir, "recipes.csv"), index=False)
    print("✅ Recipes processed.")
except Exception as e:
    print(f"❌ Failed to load RECIPIE.csv: {e}")

# === 2. Load and Merge Line Items ===
lineitems_dir = os.path.join(base_dir, "LINEITEMS")
lineitem_files = glob.glob(os.path.join(lineitems_dir, "*.csv"))

lineitem_data = []

print("\n🔄 Processing LINEITEMS...")
for file in lineitem_files:
    try:
        df = pd.read_csv(
            file,
            sep=',',  # ✅ fixed delimiter
            encoding='latin1',
            on_bad_lines='skip',
            engine='python'
        )
        filename = os.path.splitext(os.path.basename(file))[0]
        recipe_key = filename.replace("_", "/")
        df["RECIPEKEY"] = recipe_key
        lineitem_data.append(df)
    except Exception as e:
        print(f"⚠️ Error reading {file}: {e}")

if lineitem_data:
    line_items_df = pd.concat(lineitem_data, ignore_index=True)
    line_items_df.to_csv(os.path.join(output_dir, "line_items.csv"), index=False)
    print("✅ Line items merged and saved.")
else:
    print("❌ No line item files loaded successfully.")

# === 3. Load and Merge Specifications ===
spec_dir = os.path.join(base_dir, "SPECIFICATION")
spec_files = glob.glob(os.path.join(spec_dir, "*.csv"))

spec_data = []

print("\n🔄 Processing SPECIFICATIONS...")
for file in spec_files:
    try:
        df = pd.read_csv(
            file,
            sep=',',  # ✅ fixed delimiter
            encoding='latin1',
            on_bad_lines='skip',
            engine='python'
        )
        spec_id = os.path.splitext(os.path.basename(file))[0]
        df["SPECIFICATIONNR"] = spec_id
        spec_data.append(df)
    except Exception as e:
        print(f"⚠️ Error reading {file}: {e}")

if spec_data:
    specs_df = pd.concat(spec_data, ignore_index=True)
    specs_df.to_csv(os.path.join(output_dir, "specifications.csv"), index=False)
    print("✅ Specifications merged and saved.")
else:
    print("❌ No specification files loaded successfully.")

print("\n✅ All tasks completed. Output saved to:", output_dir)





# import os
# import pandas as pd
# import glob
# import csv

# # === CONFIGURATION ===
# base_dir = "PML_DATA"  # folder containing RECIPIE.csv, LINEITEMS/, SPECIFICATION/
# output_dir = "PML_OUTPUT_2"
# os.makedirs(output_dir, exist_ok=True)

# # === 1. Load Recipes ===
# try:
#     recipe_path = os.path.join(base_dir, "RECIPIE.csv")
#     recipes_df = pd.read_csv(recipe_path, sep=';', encoding='latin1', engine='python')
#     recipes_df.to_csv(os.path.join(output_dir, "recipes.csv"), index=False)
#     print("✅ Recipes processed.")
# except Exception as e:
#     print(f"❌ Failed to load RECIPIE.csv: {e}")

# # === 2. Load and Merge Line Items ===
# lineitems_dir = os.path.join(base_dir, "LINEITEMS")
# lineitem_files = glob.glob(os.path.join(lineitems_dir, "*.csv"))

# lineitem_data = []

# print("\n🔄 Processing LINEITEMS...")
# for file in lineitem_files:
#     try:
#         df = pd.read_csv(
#             file,
#             sep=';',
#             encoding='latin1',
#             quoting=csv.QUOTE_NONE,
#             on_bad_lines='skip',
#             engine='python'
#         )
#         filename = os.path.splitext(os.path.basename(file))[0]
#         recipe_key = filename.replace("_", "/")
#         df["RECIPEKEY"] = recipe_key
#         lineitem_data.append(df)
#     except Exception as e:
#         print(f"⚠️ Error reading {file}: {e}")

# if lineitem_data:
#     line_items_df = pd.concat(lineitem_data, ignore_index=True)
#     line_items_df.to_csv(os.path.join(output_dir, "line_items.csv"), index=False)
#     print("✅ Line items merged and saved.")
# else:
#     print("❌ No line item files loaded successfully.")

# # === 3. Load and Merge Specifications ===
# spec_dir = os.path.join(base_dir, "SPECIFICATION")
# spec_files = glob.glob(os.path.join(spec_dir, "*.csv"))

# spec_data = []

# print("\n🔄 Processing SPECIFICATIONS...")
# for file in spec_files:
#     try:
#         df = pd.read_csv(
#             file,
#             sep=';',
#             encoding='latin1',
#             quoting=csv.QUOTE_NONE,
#             on_bad_lines='skip',
#             engine='python'
#         )
#         spec_id = os.path.splitext(os.path.basename(file))[0]
#         df["SPECIFICATIONNR"] = spec_id
#         spec_data.append(df)
#     except Exception as e:
#         print(f"⚠️ Error reading {file}: {e}")

# if spec_data:
#     specs_df = pd.concat(spec_data, ignore_index=True)
#     specs_df.to_csv(os.path.join(output_dir, "specifications.csv"), index=False)
#     print("✅ Specifications merged and saved.")
# else:
#     print("❌ No specification files loaded successfully.")

# print("\n✅ All tasks completed. Output saved to:", output_dir)





# import os
# import pandas as pd
# import glob

# # === CONFIGURATION ===
# base_dir = "PML_DATA"  # folder that contains all the subfolders
# output_dir = "PML_OUTPUT"
# os.makedirs(output_dir, exist_ok=True)

# # === 1. Load Recipes ===
# recipe_path = os.path.join(base_dir, "RECIPIE.csv")
# recipes_df = pd.read_csv(recipe_path, sep=';', encoding='latin1')

# recipes_df.to_csv(os.path.join(output_dir, "recipes.csv"), index=False)

# # === 2. Load and Merge Line Items ===
# lineitems_dir = os.path.join(base_dir, "LINEITEMS")
# lineitem_files = glob.glob(os.path.join(lineitems_dir, "*.csv"))

# lineitem_data = []

# for file in lineitem_files:
#     try:
#         df = pd.read_csv(file, sep=';', encoding='latin1')
#         filename = os.path.splitext(os.path.basename(file))[0]
#         recipe_key = filename.replace("_", "/")
#         df["RECIPEKEY"] = recipe_key
#         lineitem_data.append(df)
#     except Exception as e:
#         print(f"Error reading {file}: {e}")

# line_items_df = pd.concat(lineitem_data, ignore_index=True)
# line_items_df.to_csv(os.path.join(output_dir, "line_items.csv"), index=False)

# # === 3. Load and Merge Specifications ===
# spec_dir = os.path.join(base_dir, "SPECIFICATION")
# spec_files = glob.glob(os.path.join(spec_dir, "*.csv"))

# spec_data = []

# for file in spec_files:
#     try:
#         df = pd.read_csv(file, sep=';', encoding='latin1')
#         spec_id = os.path.splitext(os.path.basename(file))[0]
#         df["SPECIFICATIONNR"] = spec_id
#         spec_data.append(df)
#     except Exception as e:
#         print(f"Error reading {file}: {e}")

# specs_df = pd.concat(spec_data, ignore_index=True)
# specs_df.to_csv(os.path.join(output_dir, "specifications.csv"), index=False)

# print("✅ All files processed and saved to:", output_dir)
