#!/bin/bash

# 运行 assembler_and_book_id_mapping.py
echo "Running assembler_and_book_id_mapping.py..."
/home/hadoop/anaconda3/envs/pyspark_env/bin/python assembler_and_book_id_mapping.py
if [ $? -ne 0 ]; then
    echo "assembler_and_book_id_mapping.py failed. Exiting."
    exit 1
fi

# 运行 similarity_minihash.py
echo "Running similarity_minihash.py..."
/home/hadoop/anaconda3/envs/pyspark_env/bin/python similarity_minihash.py
if [ $? -ne 0 ]; then
    echo "similarity_minihash.py failed. Exiting."
    exit 1
fi

# 运行 recommend_for_user.py
echo "Running recommend_for_user.py..."
/home/hadoop/anaconda3/envs/pyspark_env/bin/python recommend_for_user.py
if [ $? -ne 0 ]; then
    echo "recommend_for_user.py failed. Exiting."
    exit 1
fi

echo "All scripts executed successfully."
