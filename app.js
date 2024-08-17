const userinput = prompt('Enter the first number:');
const userinput2 = prompt('Enter the second number:');
const symbol = prompt('Enter the symbol (+, -, *, /):');

const num1 = parseFloat(userinput);
const num2 = parseFloat(userinput2);

if (symbol === "+") {
    console.log(num1 + num2);
} else if (symbol === "-") {
    console.log(num1 - num2);
} else if (symbol === "*") {
    console.log(num1 * num2);
} else if (symbol === "/") {
    if (num2 === 0) {
        console.log('Not divisible by zero');
    } else {
        console.log(num1 / num2);
    }
} else {
    console.log('Invalid symbol');
}
