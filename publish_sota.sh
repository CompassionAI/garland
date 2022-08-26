#!/bin/bash
cd ./translations
for f in ./*.md;
do
  echo Processing ${f}...
  pandoc ${f} --pdf-engine=xelatex -V header-includes:'\setromanfont{Jomolhari:style=Regular}' -o ${f%.md}.pdf
done

echo Zipping...
zip results.zip *.pdf

echo Uploading to S3...
aws s3 cp results.zip s3://compassionai/public/

echo Cleaning up...
rm *.pdf results.zip

echo Done!
