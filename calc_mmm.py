###################################
# find out mean, median, mode from the given array
# example: mean: average: sum/count
# example: median: sorted array, middle element if count is odd or average of middle two if count is even.
# example: mode is an lowest value element
#          in the array that occurred most.
#          if all occurred only once then 
#          the lowest item is mode.
###################################
def calc_mmm(arr):
	""" ........ calc mean ..... """
	mean = sum(arr)/len(arr)

	""" ........ calc median ..... """
	median = 0
	n = int(len(arr)/2)
	arr_s = sorted(arr)
	
	if len(arr_s)%2 == 1:
		median = arr_s[n]
	else:
		median = (arr_s[n] + arr_s[n+1]) / 2

	""" ........ calc mode ..... """
	mode = 0
	frq_map = {}
	min_val = 0xFFFFFFFF
	max_frq = 0
	for i in range(len(arr)):
		if not frq_map.__contains__(arr[i]):
			frq_map[arr[i]] = 1
		else:
			frq_map[arr[i]] += 1
		if max_frq < frq_map[arr[i]]:
			max_frq = frq_map[arr[i]]  # look for ocurring most.
		if min_val > arr[i]:
			min_val = arr[i]           # look for min value.
	if max_frq == 1:
		mode = min_val                     # lowest value as all ocurred only once.
	else:
		max_frq = 0
		for k in frq_map.keys():
			if  max_frq < frq_map[k]:
				max_frq = frq_map[k]
				min_val = k
			elif max_frq == k:
				min_val = min(min_val, k)
	mode = min_val

	return mean, median, mode

if __name__ == "__main__":
	print("-------------------------")
	arr = [1, 2, 3, 5, 7, 2, 3, 1]
	x, y, z = calc_mode(arr)
	print(arr, "mean: %.2f" %(x), " median: ", y, " mode: ", z)
	print("-------------------------")
	arr = [1, 2, 3, 5, 7, 2, 3, 1, 2]
	x, y, z = calc_mode(arr)
	print(arr, "mean: %.2f" %(x), " median: ", y, " mode: ", z)
	print("-------------------------")
	arr = [1, 2, 3, 4, 5]
	x, y, z = calc_mode(arr)
	print(arr, "mean: %.2f" %(x), " median: ", y, " mode: ", z)
	print("-------------------------")
