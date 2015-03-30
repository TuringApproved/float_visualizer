import numpy as np
import itertools as it
import copy

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

#  Some globals
max_exp_bits = 5
max_mant_bits = 6

exp_bits = 2 # initial values
mant_bits = 3


# I just create a handy function to represent integers as signed binary numbers.
# Python's implementation of the function is a little awkward to use (in this particular use case).

def bn(some_int):
    """
    returns a binary representation of an integer
    the form is +/-ddd...d where each d is a binary digit
    """
    s_binary = bin(some_int)
    s_binary = ("+" + s_binary[2:]) if s_binary[0] is not "-" else (s_binary[0]+s_binary[3:])
    return s_binary


class MyLittleFloat():
    """A float object with a sign, exponent (biased) and mantissa"""
    
    def __init__(self, sign, mantissa, exponent, bias="half", denorm=False):
        """
        sign, mantissa, exponent and bias have to be strings.
        sign can be "+" or "-",
        everything else has to be a string of 0s and 1s
        """

        self.sign = -1 if sign is "-" else 1  # I use this as a multiplier
        self.exponent = exponent
        self.bias = bn(2**(len(exponent)-1))[1:] if bias is "half" else bias
        self.mantissa = "0" + mantissa if denorm else "1" + mantissa

        self.exp_bits = len(exponent)
        self.mant_bits = len(mantissa)  # As the first digit of mantissa

        if self.mantissa[0] == 0 and int(self.exponent, 2) != 0:
            raise Exception("Exponent has to be 0 for denormalized float")

    def __add__(self, other):
        """
        This function overloads the + operator, so we can sum mylittlefloats
        """

        # the addends have the same float representation
        assert(self.mant_bits == other.mant_bits)
        assert(self.exp_bits == other.exp_bits)
        assert(self.bias == other.bias)

        # convert the addends to integers, and convert the result to bn string
        result = bn(self.sign*int(self.mantissa, 2)*2**int(self.exponent, 2) +
                    other.sign*int(other.mantissa, 2)*2**int(other.exponent, 2))

        r_sign = result[0]  # because of what bn returns (e.g. "+10")
        # mantissa has still to be modified to match right representation (could be too long or too short)
        r_mant = result[1:]

        # if r_mant is too small, add zeros before
        if len(r_mant) < (self.mant_bits+1):
            r_mant = "0"*(self.mant_bits+1)

        #  r_mant is now rounded to the right number of digits.
        #  The number of remaining digits informs the exponent
        
        if len(r_mant) != self.mant_bits+1:
            r_mant = self.round_to_even(r_mant, self.mant_bits+1)
            
        if len(r_mant)-(self.mant_bits+1) >= 0:
            r_expo = bn(len(r_mant)-(self.mant_bits+1))[1:]
            r_mant = r_mant[:(self.mant_bits+1)]  # truncate
        else:
            # I could trow an underflow exception here, but I just set to 0
            r_mant = "0"
            r_expo = "0"
        
        denorm = True if r_mant[0] == "0" else False
            
        return MyLittleFloat(r_sign, r_mant[1:], r_expo, bias=self.bias, denorm=denorm)

    def __sub__(self, other):
        """Just adds, but inverts the sign of the second addend"""

        other_copy = copy.copy(other)
        other_copy.sign = -1 if other_copy.sign == 1 else 1
        return self + other_copy

    def round_to_even(self, bin_n, rounding_digit):
        """
        With rounding digit being interpreted as the first digit of the fractional part of bin_n,
        round bin_n up to nearest number. If it's a tie, round to nearest even number.
        For example, if bin_n="1101" and you want to round it to the second digit (rounding_digit=2),
        the result becomes "10000".
        Note that the truncation has to be handled separately.
        """

        rounded = bin_n

        if (bin_n[rounding_digit] == "1") and (len(bin_n) > rounding_digit+1):
            # the fractional part is >= 0.5, and there are other digits after the rounding digit

            if (int(bin_n[rounding_digit+1:], 2) == 0) and (bin_n[rounding_digit-1] == "1"):
                # the fractional part is == 0.5 and the integer part is odd, round up to even
                rounded = bn(int(bin_n, 2) + 2**(len(rounded)-rounding_digit-1))[1:]
            elif int(bin_n[rounding_digit+1:], 2) > 0:
                # the fractional part is >0.5, round up to whatever
                rounded = bn(int(bin_n, 2) + 2**(len(rounded)-rounding_digit-1))[1:]

        if bin_n[rounding_digit] == "0":  # if the fractional part is 0, round down
            rounded = bin_n[:rounding_digit] + "".join(["0" for _ in bin_n[rounding_digit:]])

        return rounded

    def __str__(self):
        """
        print in binary scientific notation
        """
        str_to_print = "" if self.sign is 1 else "-"
        str_to_print += str(self.mantissa[0]) + "." + self.mantissa[1:] + " * 2^("
        str_to_print += bn(int(self.exponent, 2)-int(self.bias, 2)) + ")"
        return str_to_print

    def to_p_float(self):
        """
        Convert to python's usual 64 bit float.
        Nothing is lost in conversion as my_little_float representational bits
        are all smaller than their corresponding in python's float.
        """
        mantissa_float = self.sign * int(self.mantissa,2)
        mantissa_float /= float(2**self.mant_bits)
        exponent_float = 2**(int(self.exponent, 2)-int(self.bias, 2))
        return mantissa_float * exponent_float


"""
First of all, I create a function which given the number of bits in a float representation,
it creates all possible numbers in that representation.
Specifically, I want to show three different representations,
the normalized and denormalized floats, and the fixed point representation. So this functions creates all three of them.
"""

def representable_reals(exp_bits, mant_bits):
    """
    return three objects containing all representable numbers
    given number of bits in mantissa and exponential,
    - with normalized float representation
    - with denormalized float representation
    - with fixed point representation
    each of the three objects contains the numbers 
    both in python float and in MyLittleFloat form (useful for plotting of sums later in code)
    """

    possible_signs = ["-", "+"]
    possible_exponents = ["".join(str(j) for j in i) for i in it.product([0, 1], repeat=exp_bits)]
    possible_norm_mantissas = ["".join(str(j) for j in i) for i in it.product([0, 1], repeat=mant_bits)]
    possible_denorm_mantissas = ["".join(str(j) for j in i) for i in it.product([0, 1], repeat=mant_bits)]

    norm_representable_reals = {"p_floats": [], "m_l_floats": []}
    denorm_representable_reals = {"p_floats": [], "m_l_floats": []}

    bias = "half"

    # populate array of all representable real numbers with my_little_float, normalized
    for sign in possible_signs:
        for exponent in possible_exponents:
            exp_repr_list_p = []
            exp_repr_list_m = []
            for mantissa in possible_norm_mantissas:
                mlf = MyLittleFloat(sign, mantissa, exponent, bias, denorm=False)
                exp_repr_list_p.append(mlf.to_p_float())
                exp_repr_list_m.append(mlf)

            # array has two levels, for easy plotting in different colors - no other reason
            norm_representable_reals["p_floats"].append(exp_repr_list_p)
            norm_representable_reals["m_l_floats"].append(exp_repr_list_m)

    # populate array of all representable real numbers with my_little_float, denormalized
    for sign in possible_signs:
        for exponent in ["0"*exp_bits]:
            exp_repr_list_p = []
            exp_repr_list_m = []
            for mantissa in possible_denorm_mantissas:
                mlf = MyLittleFloat(sign, mantissa, exponent, bias, denorm=True)
                exp_repr_list_p.append(mlf.to_p_float())
                exp_repr_list_m.append(mlf.to_p_float())

            # array has two levels, for easy plotting in different colors - no other reason
            denorm_representable_reals["p_floats"].append(exp_repr_list_p)
            denorm_representable_reals["m_l_floats"].append(exp_repr_list_m)

    norm_representable_reals["p_floats"] = np.array(norm_representable_reals["p_floats"])
    denorm_representable_reals["p_floats"] = np.array(denorm_representable_reals["p_floats"])

    norm_representable_reals["m_l_floats"] = np.array(norm_representable_reals["m_l_floats"])
    denorm_representable_reals["m_l_floats"] = np.array(denorm_representable_reals["m_l_floats"])

    min_repr = np.min(norm_representable_reals["p_floats"])  # minimum representable reals
    max_repr = np.max(norm_representable_reals["p_floats"])  # maximum representable reals
    fixedp_representable_reals = np.linspace(min_repr, max_repr, norm_representable_reals["p_floats"].size)

    return norm_representable_reals, denorm_representable_reals, fixedp_representable_reals


# I want this simulation to be interactive, so I'm going to create a simple graphical user interface.
# The following code just specifies the layout.


""" build GUI layout """
fig = plt.figure(dpi=100, figsize=[10, 10])

n_rows = 13

#  this is where the representable numbers will be plotted
main_plot = plt.subplot2grid((n_rows, 1), (0, 0), colspan=1, rowspan=5)
main_plot.axes.get_yaxis().set_visible(False)
main_plot.get_xaxis().tick_bottom()

#  setting the position for buttons to increment and decrement number of bits for exponent
decrement_exp_bits_ax = plt.subplot2grid((n_rows, 3), (6, 0), colspan=1, rowspan=1)
exp_bits_text_ax = plt.subplot2grid((n_rows, 3), (6, 1), colspan=1, rowspan=1)
increment_exp_bits_ax = plt.subplot2grid((n_rows, 3), (6, 2), colspan=1, rowspan=1)
exp_bits_text_ax.set_axis_off()
exp_bits_text = exp_bits_text_ax.text(0.5, 0.5, "Exponent: {} bits".format(exp_bits),
                                      verticalalignment="center", horizontalalignment="center")

#  setting the position for buttons to increment and decrement number of bits for mantissa
decrement_mant_bits_ax = plt.subplot2grid((n_rows, 3), (7, 0), colspan=1, rowspan=1)
mant_bits_text_ax = plt.subplot2grid((n_rows, 3), (7, 1), colspan=1, rowspan=1)
increment_mant_bits_ax = plt.subplot2grid((n_rows, 3), (7, 2), colspan=1, rowspan=1)
mant_bits_text_ax.set_axis_off()
mant_bits_text = mant_bits_text_ax.text(0.5, 0.5, "Mantissa: {} bits".format(mant_bits),
                                        verticalalignment="center", horizontalalignment="center")

# setting the position for sliders used for navigation
x_center_ax = plt.subplot2grid((n_rows, 1), (8, 0), colspan=1, rowspan=1)
x_zoom_ax = plt.subplot2grid((n_rows, 1), (9, 0), colspan=1, rowspan=1)

# setting the position for sliders and button to do summation
a_ax = plt.subplot2grid((n_rows, 1), (10, 0), colspan=1, rowspan=1)
b_ax = plt.subplot2grid((n_rows, 1), (11, 0), colspan=1, rowspan=1)
a_ax.set_visible(False)  # they appear when the relevant button is pressed
b_ax.set_visible(False)
sum_ax = plt.subplot2grid((n_rows, 3), (12, 1), colspan=1, rowspan=1)


# Then put in the interactive bits: buttons and sliders.

def update_from_sliders(event):
    #  update plot when slider values are modified
    print s_x_center.val
    plotting_funct(exp_bits, mant_bits, s_x_center.val, s_x_zoom.val)


def button_clicked(action):
    global exp_bits, mant_bits, do_sum

    if action == "e+":
        if exp_bits < max_exp_bits:
            exp_bits += 1
        exp_bits_text.set_text("Exponent: {} bits".format(exp_bits))

    elif action == "e-":
        if exp_bits > 1:
            exp_bits -= 1
        exp_bits_text.set_text("Exponent: {} bits".format(exp_bits))

    elif action == "m+":
        if mant_bits < max_mant_bits:
            mant_bits += 1
        mant_bits_text.set_text("Mantissa: {} bits".format(mant_bits))

    elif action == "m-":
        if mant_bits > 0:
            mant_bits -= 1
        mant_bits_text.set_text("Mantissa: {} bits".format(mant_bits))

    elif action == "sum":
        a_ax.set_visible(not a_ax.get_visible())
        b_ax.set_visible(not b_ax.get_visible())
        b_sum.label.set_text("Dectivate sum" if not do_sum else "Activate sum")
        do_sum = not do_sum

    #  refresh plot every time something is modified (a button is pressed)
    plotting_funct(exp_bits, mant_bits, s_x_center.val, s_x_zoom.val)


# create Buttons and on_clicked functions
b_exp_bits_plus = Button(increment_exp_bits_ax, 'exp bits +')
b_exp_bits_plus.on_clicked(lambda x: button_clicked("e+"))

b_exp_bits_minus = Button(decrement_exp_bits_ax, 'exp bits -')
b_exp_bits_minus.on_clicked(lambda x: button_clicked("e-"))

b_mant_bits_plus = Button(increment_mant_bits_ax, 'mant bits +')
b_mant_bits_plus.on_clicked(lambda x: button_clicked("m+"))

b_mant_bits_minus = Button(decrement_mant_bits_ax, 'mant bits -')
b_mant_bits_minus.on_clicked(lambda x: button_clicked("m-"))

# create sliders for navigation
s_x_center = Slider(x_center_ax, 'center', -1, 1, valfmt='%1.1f', valinit=0, color="black")
s_x_center.valtext.set_visible(False)
s_x_center.on_changed(update_from_sliders)

s_x_zoom = Slider(x_zoom_ax, 'zoom', -1, 0.99999, valfmt='%1.1f', valinit=0, color="black", closedmax=True)
s_x_zoom.valtext.set_visible(False)
s_x_zoom.on_changed(update_from_sliders)

# create sliders for summation
s_a = Slider(a_ax, 'a', -1, 1, valfmt='%.3f', valinit=-1, color="black")
s_a.valtext.set_visible(False)
s_a.on_changed(update_from_sliders)

s_b = Slider(b_ax, 'b', -1, 1, valfmt='%.3f', valinit=1, color="black")
s_b.valtext.set_visible(False)
s_b.on_changed(update_from_sliders)

# create button to activate/deactivate summation
b_sum = Button(sum_ax, "Activate sum")
b_sum.on_clicked(lambda x: button_clicked("sum"))

do_sum = False


# Finally, a function to plot our numbers.
# In the visualization it's possible to change the number of bits for mantissa and exponential
#  and look at the numbers that can be represented.

def plotting_funct(exp_bits, mant_bits, x_center, zoom):

    main_plot.clear()  # to refresh

    norm_repr_r, denorm_repr_r, fixedp_repr_r = representable_reals(exp_bits, mant_bits)

    for exponent_list in norm_repr_r["p_floats"]:
        main_plot.plot(exponent_list, [0]*len(exponent_list), linestyle="", marker="|", ms=20, mew=3)
    norm_fp_text = main_plot.text(0, -0.7, "normalized floating point", size=20,
                                  horizontalalignment='center')

    x_min = np.min(norm_repr_r["p_floats"])  # minimum representable real
    x_max = np.max(norm_repr_r["p_floats"])  # maximum representable real

    x_range = float(x_max - x_min)
    x_min -= x_range/10
    x_max += x_range/10
    main_plot.axis([x_min*(1-zoom) - x_center*x_min, x_max*(1-zoom) + x_center*x_max, -0.8, 0.6])

    if not do_sum:
        for d_exponent_list in denorm_repr_r["p_floats"]:
            main_plot.plot(d_exponent_list, [0.5]*len(d_exponent_list),
                           linestyle="", marker="|", ms=20, mew=3)
        main_plot.text(0, 0.3, "denormalized floating point", size=20, horizontalalignment='center')

        main_plot.plot(fixedp_repr_r, [-0.5]*fixedp_repr_r.size, linestyle="", marker="|", ms=20, mew=3)
        main_plot.text(0, -0.7, "fixed point", size=20, horizontalalignment='center')

        norm_fp_text.set_position((0, -0.2))

    else:
        nearest_1 = np.abs(norm_repr_r["p_floats"] - s_a.val*x_max).argmin()
        nearest_2 = np.abs(norm_repr_r["p_floats"] - s_b.val*x_max).argmin()

        n1_p = norm_repr_r["p_floats"].flat[nearest_1]
        n2_p = norm_repr_r["p_floats"].flat[nearest_2]
        n1_m = norm_repr_r["m_l_floats"].flat[nearest_1]
        n2_m = norm_repr_r["m_l_floats"].flat[nearest_2]

        actual_sum = n1_p + n2_p
        repr_sum = (n1_m + n2_m).to_p_float()

        main_plot.annotate("a", size=20, xy=(n1_p, 0.05), xytext=(n1_p, 0.1),
                           arrowprops=dict(facecolor='black', shrink=0.05))
        main_plot.annotate("b", size=20, xy=(n2_p, 0.05), xytext=(n2_p, 0.1),
                           arrowprops=dict(facecolor='black', shrink=0.05))

        main_plot.annotate("Actual sum: {}".format(actual_sum), size=20, horizontalalignment="center",
                           xy=(actual_sum, 0), xytext=(actual_sum, 0.4),
                           arrowprops=dict(facecolor='black', shrink=0.05))
        main_plot.annotate("Representable sum: {}".format(repr_sum), size=20, horizontalalignment="center",
                           xy=(repr_sum, 0), xytext=(repr_sum, -0.4),
                           arrowprops=dict(facecolor='black', shrink=0.05))
    plt.draw()


plotting_funct(exp_bits, mant_bits, 0, 0.0)
plt.show()


# The visualization makes it really easy to visualize the role of the exponential
# and the mantissa in the representation.
# Also, you can press the sum button at the bottom to add two floats.
# The a and b sliders let you choose the floats, and two sums are displayed.
# One is the sum obtained by using the precision of the float you're adding.
# The second is the exact result computed with python floats.
# The result is exact because python's floating point representation (well, the double precision IEEE754)
# has higher precision than the floats we are using.
# Note how for example adding a big number and a small one easily leads to **error in the representation**.