#! python2
# coding: utf-8

def main():
    data = [(1,1),(1,2),(3,3),(1,1),(1,3)]
    hashMapA = {}
    hashMapB = {}
    hashMapAB = {}
    countA = 0
    countB = 0
    countAB = 0
    for item in data:
        # key a
        if not hashMapA.get(item[0]):
            hashMapA[item[0]] = 1
        else:
            hashMapA[item[0]] += 1
        # key b
        if not hashMapB.get(item[1]):
            hashMapB[item[1]] = 1
        else:
            hashMapB[item[1]] += 1
        # key ab
        if not hashMapAB.get(item):
            hashMapAB[item] = 1
        else:
            hashMapAB[item] += 1

    for value in hashMapA.values():
        if value > 1:
            countA += value*(value-1)/2

    for value in hashMapB.values():
        if value > 1:
            countB += value*(value-1)/2

    for value in hashMapAB.values():
        if value > 1:
            countAB += value*(value-1)/2

    print countA
    print countB
    print countAB
    print countA + countB - countAB

if __name__ == '__main__':
    main()