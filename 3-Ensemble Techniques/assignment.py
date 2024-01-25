# Write a function to find the longest common prefix string amongst an array of strings.
# If there is no common prefix, return an empty string "".
# Example 1:
# Input: ["flower","flow","flight"]
# Output: "fl"
# Example 2:
# Input: ["dog","racecar","car"]
# Output: ""
# Explanation: There is no common prefix among the input strings.
# Note: All given inputs are in lowercase letters a-z.
# Solution:
# 1. Find the shortest string in the list
# 2. Iterate through the shortest string and compare each character with the other strings
# 3. If the character is not present in any of the other strings, return the string till that character
# 4. If the character is present in all the other strings, return the shortest string
# 5. If the shortest string is empty, return empty string
# 6. If the list is empty, return empty string
# 7. If the list has only one string, return the string
# 8. If the list has only one string and it is empty, return empty string
# 9. If the list has only one string and it is not empty, return the string
# 10. If the list has more than one string and all the strings are empty, return empty string
# 11. If the list has more than one string and all the strings are not empty, return the shortest string
# 12. If the list has more than one string and some strings are empty, return empty string
# 13. If the list has more than one string and some strings are not empty, return the shortest string
# 14. If the list has more than one string and all the strings are same, return the shortest string
# 15. If the list has more than one string and all the strings are not same, return empty string
# 16. If the list has more than one string and all the strings are same and empty, return empty string
# 17. If the list has more than one string and all the strings are same and not empty, return the shortest string
# 18. If the list has more than one string and some strings are same and some strings are not same, return empty string
# 19. If the list has more than one string and some strings are same and some strings are not same and some strings are empty, return empty string

def longestCommonPrefix(strs):
    """
    :type strs: List[str]
    :rtype: str
    """
    if len(strs) == 0:
        return ""
    elif len(strs) == 1:
        if strs[0] == "":
            return ""
        else:
            return strs[0]
    else:
        shortest_string = min(strs, key=len)
        if shortest_string == "":
            return ""
        else:
            for i in range(len(shortest_string)):
                for j in range(len(strs)):
                    if shortest_string[i] != strs[j][i]:
                        return shortest_string[:i]
            return shortest_string
        
# Test Cases:
# 1. longestCommonPrefix(["flower","flow","flight"]) -> "fl"
output = longestCommonPrefix(["flower","flow","flight"])

print(output)

