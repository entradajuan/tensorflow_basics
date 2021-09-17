list1 = ["halo", [1,2,3], 4]
print(list1[1])

list1.append("Blanco")

for e in list1:
  print(e)

#----------------------------------------------

list2 = ["AAA", "BBB", "CCC"]

def strip(word):
  return word.lower()

#list2 = [strip(w) for w in list2]
list2 = [(lambda w : w.lower())(w) for w in list2]

print(list2)

first = lambda w : w[:1]
list2 = map(first, list2)

#for e in list2:
#  print(e)

print(list2)
sol = list(list2)
print(sol)
  





