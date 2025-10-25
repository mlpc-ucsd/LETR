#!/bin/bash

dir2bin() {
    case "$1" in
    "VISIGRAPP_TRAIN/dataset0" )
        echo 0
        ;;
    "VISIGRAPP_TRAIN/dataset1")
        echo 1
        ;;
    "VISIGRAPP_TRAIN/dataset2")
        echo 0
        ;;
    "VISIGRAPP_TRAIN/dataset3")
        echo 2
        ;;
    "VISIGRAPP_TRAIN/dataset4")
        echo 2
        ;;
    "VISIGRAPP_TRAIN/ElavatedGrayBox")
        echo 0
        ;;
    "VISIGRAPP_TRAIN/ElevatedGreyBox")
        echo 0
        ;;
    "VISIGRAPP_TRAIN/ElevatedGreyFullBeer")
        echo 0
        ;;
    "VISIGRAPP_TRAIN/FirstRealSet")
        echo 3
        ;;
    "VISIGRAPP_TRAIN/GoldBinAdditional")
        echo 3
        ;;
    "VISIGRAPP_TRAIN/GrayBoxPad")
        echo 0
        ;;
    "VISIGRAPP_TRAIN/LargeWoodenBoxDynamic")
        echo 2
        ;;
    "VISIGRAPP_TRAIN/LargeWoodenBoxStatic")
        echo 2
        ;;
    "VISIGRAPP_TRAIN/ShallowGreyBox")
        echo 0
        ;;
    "VISIGRAPP_TRAIN/SmalGreyBasket")
        echo 4
        ;;
    "VISIGRAPP_TRAIN/SmallGoldenBox")
        echo 3
        ;;
    "VISIGRAPP_TRAIN/SmallWhiteBasket")
        echo 1
        ;;
    "VISIGRAPP_TRAIN/synth_dataset5_random_origin")
        echo 5
        ;;
    "VISIGRAPP_TRAIN/synth_dataset6_random_origin")
        echo 6
        ;;
    "VISIGRAPP_TRAIN/synth_dataset7_random_origin")
        echo 5
        ;;
    "VISIGRAPP_TRAIN/synth_dataset8_random_origin")
        echo 5
        ;;
    "VISIGRAPP_TEST/TestBin")
        echo 3
        ;;
    "VISIGRAPP_TEST/TestCarton")
        echo 7
        ;;
    "VISIGRAPP_TEST/TestGold")
        echo 3
        ;;
    "VISIGRAPP_TEST/TestGray")
        echo 0
        ;;
    "VISIGRAPP_TEST/TestSynth")
        echo 5
        ;;
    *)
        echo -1
        ;;
    esac
}

lines_dir="$1"

if [ ! -d "$lines_dir" ] ; then
    echo "Lines directory '$lines_dir' not found"
    exit
fi

# using base64 to handle spaces in path
for d in $(find data/VISIGRAPP_TRAIN data/VISIGRAPP_TEST -mindepth 1 -maxdepth 1 -type d -print0 | xargs --null -I{} sh -c "echo {} | base64") ; do
    dir="$(echo "$d" | base64 -d)"
    bin="bin$(dir2bin "$dir")"

    [ $bin == "bin-1" ] && continue

    link="${dir}/bin_lines.txt"
    file="${lines_dir}/$bin.txt"

    echo "linking $(realpath $file) to $link"
    ln -rsf "$file" "$link"
done


