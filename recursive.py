def swap(a, b):
    if a < b:
        return a, b
    else:
        return b, a


def multiply_recurse(a, b, memo):
    if not (a, b) in memo:
        if a == 1:
            memo[(a, b)] = b
        elif a % 2 == 0:
            memo[(a, b)] = multiply_recurse(a // 2, b, memo) << 1
        else:
            memo[(a, b)] = multiply_recurse(a - 1, b, memo) + b
    return memo[(a, b)]


def multiply(a, b):
    a, b = swap(a, b)
    return multiply_recurse(a, b, {})


def permutation_recurse(s, chars, all_perms):
    num_possibles = len(chars)
    if num_possibles == 0:
        all_perms.append(s)
        return
    for curr_char in chars:
        new_chars = chars.copy()
        new_chars.remove(curr_char)
        permutation_recurse(s + curr_char, new_chars, all_perms)


def permutations(s):
    all_chars = set([c for c in s])
    all_perms = []
    permutation_recurse("", all_chars, all_perms)
    return all_perms


QUARTER = 25
DIME = 10
NICKEL = 5
PENNIE = 1


def coins_rec(target, current_value, possible_coins):
    if current_value == target:
        return 1

    num_reps = 0
    biggest_possible_coin = possible_coins[0]
    if current_value + biggest_possible_coin <= target:
        num_reps += coins_rec(target, current_value + biggest_possible_coin, possible_coins)
    if len(possible_coins) > 1:
        num_reps += coins_rec(target, current_value, possible_coins[1:])

    return num_reps


def coins(target):
    return coins_rec(target, 0, [QUARTER, DIME, NICKEL, PENNIE])


print(coins(10))


