def longest(nums):
    if not nums:
        return []
    n = len(nums)
    dp = [1]*n
    prev = [-1]*n
    for i in range(n):
        for j in range(i):
            if nums[j]>nums[i] and dp[j]+1>dp[i]:
                dp[i]=dp[j]+1
                prev[i] = j
    
    max_l = max(dp)
    idx = dp.index(max_l)

    lis = []
    while idx != -1:
        lis.append(nums[idx])
        idx = prev[idx]
    lis = lis[::-1]
    return lis

print(longest([8,5,4,9,8,10,7,8,5]))