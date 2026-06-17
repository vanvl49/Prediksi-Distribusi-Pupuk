import os
import sys

# Add root and src to path
root_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(root_dir, 'src')
sys.path.insert(0, root_dir)
sys.path.insert(0, src_dir)

print("=== Testing Training Pipeline ===")
print(f"Root dir: {root_dir}")
print(f"Src dir: {src_dir}")
print()

try:
    # Step 1: Preprocessing
    print("Step 1: Preprocessing...")
    from data_preprocessing import preprocess_data
    train_scaled, test_scaled, scaler, freq_map = preprocess_data(os.path.join(root_dir, 'data', 'Distribusi_Pupuk_Jatim_2023-2025.csv'))
    print("✅ Preprocessing complete!")
    
    # Step 2: Modeling
    print("\nStep 2: Modeling...")
    from modeling import main as run_modeling
    # The main() function should handle running the pipeline
    # But wait, let's run it from within the src directory context?
    os.chdir(root_dir)
    
    print("Done!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    print(traceback.format_exc())
