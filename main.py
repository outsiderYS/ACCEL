

list = [1.2, 2.4, 3.6]
f = open("test.txt", "w")
for i in list:
    s = str(i) + "\n"
    f.write(s)
f.close()

f = open("test.txt", "r")
list = f.read()
print(list)