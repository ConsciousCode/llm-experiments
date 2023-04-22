'''
Implementation of Totem in Python
https://github.com/bwatts/Totem

add_memory -> log, database
completion -> print, 
'''

def binary_search_offset(haystack, needle):
	left, right = 0, len(haystack)
	while left < right:
		mid = (left + right) // 2
		if haystack[mid].id == needle.id:
			return mid
		elif haystack[mid].id < needle.id:
			left = mid + 1
		else:
			right = mid
	return left

class Timeline:
	def __init__(self):
		self.events = []
	
	def add(self, event):
		offset = binary_search_offset(event.id, self.events)
		
		
		def binary_search(needle_id, haystack):
		low = 0
		high = len(haystack) - 1

		while low <= high:
			mid = (low + high) // 2
			if haystack[mid].id == needle_id:
				return mid
			elif haystack[mid].id < needle_id:
				low = mid + 1
			else:
				high = mid - 1

		return -1

		self.events.append(event)

class Event:
	def __init__(self, reason):
		self.reason = reason

class Observer:
	pass

class D:
	pass