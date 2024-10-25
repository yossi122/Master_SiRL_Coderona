
echo "------ Zipping Statistics... ------"
if [ $# -eq 0 ] ; then
    echo "ERROR! please provide one argument which is the name of the directory holding the run output"
    exit 1
fi

if [ ! -d "outputs/$1" ]; then
  echo "ERROR! directory" + "outputs/"$1 + " DO NOT EXIST!!"
  exit 2
fi

pushd `pwd`
cd "outputs"
pwd
zip -r ../statistics_csv_"$1".zip "$1" -i \*daily_delta.csv \*amit_graph_integral.csv \*amit_graph_daily.csv \*amit_graph\*.csv \*r0\*.csv
popd
pwd
echo "------ Zipping Statistics - COMPLETE (statistics_csv.zip) ------"
