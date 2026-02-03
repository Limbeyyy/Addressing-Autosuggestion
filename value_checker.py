# unique_values_checker.py

def read_file(file_path):
    """
    Reads a text file and returns a list of lines, stripped of extra spaces.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines

def find_unique_values(file1_lines, file2_lines):
    """
    Finds unique values in each file and common values.
    """
    set1 = set(file1_lines)
    set2 = set(file2_lines)
    
    unique_in_file1 = set1 - set2
    unique_in_file2 = set2 - set1
    common_values = set1 & set2
    
    return unique_in_file1, unique_in_file2, common_values

def main():
    # Hardcoded file paths
    file1_path = r"C:\Users\Yukesh Dhakal\OneDrive\Documents\Desktop\kataho\nepali_kataho_code.txt"
    file2_path = r"C:\Users\Yukesh Dhakal\OneDrive\Documents\Desktop\kataho\purano nepali_kataho_code.txt"

    file1_lines = read_file(file1_path)
    file2_lines = read_file(file2_path)

    unique1, unique2, common = find_unique_values(file1_lines, file2_lines)

    output_file = "new_filter.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Unique values in first file:\n")
        for val in unique1:
            f.write(val + "\n")
        
        f.write("\nUnique values in second file:\n")
        for val in unique2:
            f.write(val + "\n")
        
        f.write("\nCommon values in both files:\n")
        for val in common:
            f.write(val + "\n")

    print(f"\nOutput saved to {output_file}")

if __name__ == "__main__":
    main()
