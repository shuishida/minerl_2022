#!/bin/sh
version_tag=`cat version.txt`
echo "$version_tag"
git tag -am "$version_tag" $version_tag
git push aicrowd $version_tag
