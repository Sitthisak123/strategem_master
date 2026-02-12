import csv
import requests
from bs4 import BeautifulSoup
import re

def fetch_and_extract_stratagems(url, output_filename="stratagems.csv"):
    try:
        # Fetch the webpage content
        response = requests.get(url)
        response.raise_for_status()
        html_content = response.text
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        tables = soup.find_all('table', class_='wikitable')
        
        extracted_data = []
        global_index = 1
        
        # Mapping for the directions as requested
        direction_map = {
            'LEFT': '1',
            'UP': '2',
            'RIGHT': '3',
            'DOWN': '4'
        }
        
        for table in tables:
            rows = table.find_all('tr')
            
            for row in rows:
                cells = row.find_all(['td', 'th'])
                
                # Determine cell offsets based on row structure (Category header vs Normal row)
                is_category_row = (cells[0].name == 'th')
                
                if is_category_row:
                    if len(cells) < 4: continue
                    # icon_cell = cells[1] # Not used currently
                    name_cell = cells[2]
                    code_cell = cells[3]
                else:
                    if len(cells) < 3: continue
                    # icon_cell = cells[0] # Not used currently
                    name_cell = cells[1]
                    code_cell = cells[2]

                # Verify it's a stratagem row by checking for arrow images
                arrow_imgs = code_cell.find_all('img')
                if not arrow_imgs:
                    continue

                # --- EXTRACT NAME ---
                name = name_cell.get_text(strip=True)

                # --- EXTRACT CODE ---
                code_sequence = []
                for img in arrow_imgs:
                    alt_text = img.get('alt', '')
                    # Regex to capture direction from "Stratagem Arrow Down.svg" -> "Down"
                    match = re.search(r'Arrow\s(\w+)', alt_text, re.IGNORECASE)
                    if match:
                        direction = match.group(1).upper()
                        # Map direction to number
                        if direction in direction_map:
                            code_sequence.append(direction_map[direction])
                
                # Join numbers without space
                code_string = "".join(code_sequence)

                if name and code_string:
                    data = {
                        "Index": global_index,
                        "Icon": "",
                        "Name": name,
                        "Code": code_string
                    }
                    extracted_data.append(data)
                    global_index += 1

        # Write to CSV
        headers = ["Index", "Icon", "Name", "Code"]
        
        with open(output_filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            writer.writeheader()
            writer.writerows(extracted_data)
            
        print(f"Successfully exported {len(extracted_data)} stratagems to '{output_filename}'.")
        print(f"Example code format: {extracted_data[0]['Code'] if extracted_data else 'N/A'}")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    target_url = "https://helldivers.wiki.gg/wiki/Stratagems"
    fetch_and_extract_stratagems(target_url)