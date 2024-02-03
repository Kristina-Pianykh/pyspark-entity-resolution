#!/usr/bin/env bash

# usage() {
# 	echo "Usage: $0 --url <url> --source-path <source-path> --dest-path <dest-path>"
# 	exit 1
# }

year_range=$1


SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DATA_DIR=$SCRIPT_DIR/../data

dblp_url="https://lfs.aminer.cn/lab-datasets/citation/dblp.v8.tgz"
acm_url="https://lfs.aminer.cn/lab-datasets/citation/citation-acm-v8.txt.tgz"

dblp_raw="dblp.txt"
acm_raw="citation-acm-v8.txt"

# Check if the files already exist
# else download and extract them
download_and_extract() {

	if [ "$#" -ne 2 ]; then
      echo "Error: Two arguments are required."
      return 1
  fi

	url=$1
	raw=$2

	if [ ! -f "$DATA_DIR/$raw" ]; then
		echo "Downloading $url"
		curl -o $DATA_DIR/${raw}.tgz $url
		tar -xvzf $DATA_DIR/${raw}.tgz -C $DATA_DIR
	else
		echo "$DATA_DIR/$raw already exists."
	fi
}

echo ""
echo "Downloading and extracting data files to $DATA_DIR"
download_and_extract $dblp_url $dblp_raw
download_and_extract $acm_url $acm_raw
# wait

echo ""
echo "Cleaning data files..."
echo "Output files are written to $DATA_DIR/DBLP_1995_2004 and $DATA_DIR/ACM_1995_2004"
python $SCRIPT_DIR/prepare_data.py --raw $DATA_DIR/$dblp_raw --dest $DATA_DIR/DBLP_1995_2004 &
python $SCRIPT_DIR/prepare_data.py --raw $DATA_DIR/$acm_raw --dest $DATA_DIR/ACM_1995_2004
wait

echo ""
echo "Run matching algorithm..."
if [ -z "$year_range" ]; then
	echo "Output files are written to $DATA_DIR/duplicate_candidates/full"
	duplicate_candidates_path=$DATA_DIR/duplicate_candidates/full/
	matched_entities_path=$DATA_DIR/matched_entities/full/
	python $SCRIPT_DIR/match.py --dblp_path $DATA_DIR/DBLP_1995_2004 --acm_path $DATA_DIR/ACM_1995_2004 --dest $DATA_DIR/duplicate_candidates/full/
else
	echo "Output files are written to $DATA_DIR/duplicate_candidates/blocked"
	duplicate_candidates_path=$DATA_DIR/duplicate_candidates/blocked/
	matched_entities_path=$DATA_DIR/matched_entities/blocked/
	python $SCRIPT_DIR/match.py --dblp_path $DATA_DIR/DBLP_1995_2004 --acm_path $DATA_DIR/ACM_1995_2004 --dest $DATA_DIR/duplicate_candidates/blocked/ --year_range $year_range
fi

if [ $? -ne 0 ]; then
	echo "Error: Matching algorithm failed."
	exit 1
fi

echo ""
echo "Generating the graph..."
echo "Output files are written to $matched_entities_path"
python $SCRIPT_DIR/create_graph.py --duplicates_path $duplicate_candidates_path --raw_dblp data/DBLP_1995_2004 --raw_acm data/ACM_1995_2004 --dest $matched_entities_path

echo ""
echo "Cleaning the tmp files $DATA_DIR/$dblp_raw and $DATA_DIR/$acm_raw"
rm $DATA_DIR/${dblp_raw}.tgz > /dev/null 2>&1
rm $DATA_DIR/${acm_raw}.tgz > /dev/null 2>&1
# rm -rf $DATA_DIR/DBLP_1995_2004
# rm -rf $DATA_DIR/ACM_1995_2004
