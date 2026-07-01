#!/bin/bash


# === activate conda environment（with python、DetectNodes、StitchNodes）===
conda activate pangu
method="VI"
DIR="${method}/nc/2023-05"
prefix_dir="ANONYMOUS_PATH/"



track () {

    IN_FILE="$1"


    BASENAME="${IN_FILE#${prefix_dir}${DIR}/}"            

    TMP_FILE="${prefix_dir}${DIR}/${BASENAME}"      
    NODE_FILE="nodes/${BASENAME%.nc}"
    TRACKS_FILE="./tracks/${BASENAME%.nc}.csv"

    if [ ! -f "$TRACKS_FILE" ]; then

        echo "Processing $IN_FILE"


        DetectNodes \
            --in_data "$TMP_FILE" \
            --out "$NODE_FILE" \
            --searchbymin "msl" \
            --closedcontourcmd "msl,200.0,5.5,0;_DIFF(z300,z500),-58.8,6.5,1.0"\
            --mergedist 6.0 \
            --outputcmd "msl,min,0;_VECMAG(u10,v10),max,2"

        StitchNodes \
            --in "$NODE_FILE" \
            --out "$TRACKS_FILE" \
            --in_fmt "lon,lat,slp,wind10" \
            --range 8.0 \
            --mintime "12h" \
            --threshold "wind10,>=,10.0,2;lat,<=,50.0,1;lat,>=,-50.0,1" \
            --out_file_format "csv"

    else
        echo "Already tracked $IN_FILE"
    fi
}

for f in ${prefix_dir}${DIR}/*.nc; do
    start=$(date +%s)
    track "$f"
    end=$(date +%s)
    echo "$((end - start))"
done
