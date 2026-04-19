import os

import json

from tqdm import tqdm


def generate_jd_json(source_folders, output_filename="jd_dataset_final.json"):

    """

    Folders scan karke raw_jd.txt aur enhanced_job_description.md ko 

    consolidated JSON format (Input/Output pair) mein convert karti hai.

    """

    dataset = []

    skipped_count = 0

    success_count = 0


    print(f"Starting preprocessing for folders: {source_folders}")


    for base_path in source_folders:

        if not os.path.exists(base_path):

            print(f"Directory not found: {base_path}")

            continue


        # Subfolders list (numerical folders 26, 27, 28...)

        subfolders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

        

        for folder_name in tqdm(subfolders, desc=f"Processing {base_path}"):

            folder_dir = os.path.join(base_path, folder_name)

            

            raw_path = os.path.join(folder_dir, "raw_jd.txt")

            enhanced_path = os.path.join(folder_dir, "enhanced_job_description.md")


            if os.path.exists(raw_path) and os.path.exists(enhanced_path):

                try:

                    with open(raw_path, 'r', encoding='utf-8') as f:

                        raw_text = f.read().strip()

                    

                    with open(enhanced_path, 'r', encoding='utf-8') as f:

                        enhanced_text = f.read().strip()


                    # Direct mapping: Input (Raw) -> Output (Enhanced)

                    entry = {

                        "input": raw_text,

                        "output": enhanced_text

                    }

                    dataset.append(entry)

                    success_count += 1

                except Exception as e:

                    print(f"Error reading folder {folder_name}: {e}")

                    skipped_count += 1

            else:

                skipped_count += 1


    with open(output_filename, 'w', encoding='utf-8') as out_file:

        json.dump(dataset, out_file, indent=4, ensure_ascii=False)


    print("\n--- Summary ---")

    print(f"Total entries: {success_count} | Skipped: {skipped_count}")


if __name__ == "__main__":

    # Apne paths yahan update karein

    dirs_to_process = ["./dataset", "./output"] 

    generate_jd_json(dirs_to_process)
