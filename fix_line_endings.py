import sys

def convert_line_endings(file_path):
    # Read the file with universal newline support and utf-8 encoding
    with open(file_path, 'r', newline=None, encoding='utf-8') as f:
        content = f.read()

    # Write the file back with LF line endings and utf-8 encoding
    with open(file_path, 'w', newline='\n', encoding='utf-8') as f:
        f.write(content)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python fix_line_endings.py <file1> <file2> ...")
        sys.exit(1)
    
    for file_path in sys.argv[1:]:
        print(f"Converting line endings for {file_path}")
        convert_line_endings(file_path)
        print(f"Done.") 