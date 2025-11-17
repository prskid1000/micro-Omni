"""Test script to download one book and verify passage counting logic"""
import requests
import os

def download_and_test_book(book_id=1):
    """Download a single book and test the passage counting logic"""
    base_url = "https://www.gutenberg.org/files"
    
    print(f"Testing book ID: {book_id}")
    print("="*60)
    
    # Try different file formats
    for suffix in ['-0.txt', '-8.txt', '.txt']:
        url = f"{base_url}/{book_id}/{book_id}{suffix}"
        print(f"Trying: {url}")
        try:
            response = requests.get(url, timeout=30, stream=True)
            if response.status_code == 200:
                print(f"✓ Successfully downloaded from {url}")
                
                # Read and process book
                content = response.text
                print(f"Total content length: {len(content):,} characters")
                
                # Remove Project Gutenberg headers/footers
                lines = content.split('\n')
                start_idx_text = 0
                end_idx_text = len(lines)
                
                # Find start (skip header)
                for i, line in enumerate(lines):
                    if 'START OF THIS PROJECT GUTENBERG' in line.upper() or 'START OF THE PROJECT GUTENBERG' in line.upper():
                        start_idx_text = i + 1
                        break
                
                # Find end (skip footer)
                for i in range(len(lines)-1, -1, -1):
                    if 'END OF THIS PROJECT GUTENBERG' in lines[i].upper() or 'END OF THE PROJECT GUTENBERG' in lines[i].upper():
                        end_idx_text = i
                        break
                
                book_text = '\n'.join(lines[start_idx_text:end_idx_text])
                print(f"Book text length (after header/footer removal): {len(book_text):,} characters")
                print(f"Lines in book: {len(book_text.split(chr(10))):,}")
                
                # Count passages by double newlines
                all_splits = book_text.split('\n\n')
                print(f"\nTotal splits by \\n\\n: {len(all_splits)}")
                
                # Filter out empty passages
                passages = [p.strip() for p in all_splits if p.strip()]
                passage_count = len(passages)
                print(f"Non-empty passages: {passage_count}")
                print(f"Empty passages filtered: {len(all_splits) - passage_count}")
                
                # Show first few passages
                print(f"\nFirst 5 passages (first 100 chars each):")
                for i, p in enumerate(passages[:5]):
                    print(f"  Passage {i+1}: {p[:100]}...")
                
                # Show passage length distribution
                if passages:
                    lengths = [len(p) for p in passages]
                    print(f"\nPassage length stats:")
                    print(f"  Min: {min(lengths)} chars")
                    print(f"  Max: {max(lengths)} chars")
                    print(f"  Average: {sum(lengths)/len(lengths):.1f} chars")
                    print(f"  Total characters in passages: {sum(lengths):,}")
                
                # Verify: does the book text match when we join passages?
                reconstructed = '\n\n'.join(passages)
                print(f"\nVerification:")
                print(f"  Original book text length: {len(book_text):,}")
                print(f"  Reconstructed from passages: {len(reconstructed):,}")
                print(f"  Match: {book_text.strip() == reconstructed.strip()}")
                
                # Write to test file
                test_file = "data/text/test_book.txt"
                os.makedirs(os.path.dirname(test_file), exist_ok=True)
                with open(test_file, 'w', encoding='utf-8') as f:
                    f.write(book_text + '\n\n')
                print(f"\n✓ Written to {test_file}")
                print(f"  File size: {os.path.getsize(test_file):,} bytes")
                
                return True
        except Exception as e:
            print(f"✗ Failed: {e}")
            continue
    
    print("✗ Could not download book")
    return False

if __name__ == "__main__":
    # Test with a well-known book (Alice in Wonderland is usually book ID 11)
    download_and_test_book(book_id=11)

