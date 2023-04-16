IFS=',' read -ra selectedbranch_array <<< "$selectedbranch"
# Read the last used parameter from the cache
if [ ! -f branchselected ]
then
   echo  "No last used parameter branchselected not  found"
else
    last_parameter=$(cat  branchselected)
    echo "Last used parameter: $last_parameter"
fi
# Find the index of the last used parameter in the array
if [ -n "$last_parameter" ]
then
    for i in "${!selectedbranch_array[@]}"; do
        if [[ "${selectedbranch_array[$i]}" == "$last_parameter" ]]; then
            last_index=$i
            break
        fi
    done
fi
# Print unique items from the list
echo "Unique items in the list:"
echo "${selectedbranch}" | tr ',' '\n' | sort -u
# Get the next item in the list
if [ -n "$last_index" ]
then
    next_index=$((last_index + 1))
    if [ "$next_index" -eq "${#selectedbranch_array[@]}" ]
    then
        next_index=0
    fi
else
    next_index=0
fi
next_parameter="${selectedbranch_array[$next_index]}"
echo "Next parameter: $next_parameter"
# Write the next parameter to the cache file
echo "$next_parameter" > last_parameter.txt
# Set the param output for the next step
echo "param=$next_parameter" >> $GITHUB_OUTPUT
echo $next_parameter > branchselected
