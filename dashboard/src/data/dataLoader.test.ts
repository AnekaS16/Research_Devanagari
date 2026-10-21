import { parseNullableNumber } from './dataLoader';

function assertEqual<T>(actual: T, expected: T, msg: string): void {
  if (actual !== expected) {
    throw new Error(`FAIL: ${msg} (expected: ${expected}, got: ${actual})`);
  }
}

// Test 1: Empty string returns null
assertEqual(parseNullableNumber(''), null, 'Empty string must return null');

// Test 2: Whitespace-only string returns null
assertEqual(parseNullableNumber('   '), null, 'Whitespace-only string must return null');
assertEqual(parseNullableNumber('\t\n  '), null, 'Whitespace with tab/newline must return null');

// Test 3: Valid zero returns 0 (not null, not converted to false)
assertEqual(parseNullableNumber('0'), 0, 'String "0" must return numeric 0');
assertEqual(parseNullableNumber('0.0'), 0, 'String "0.0" must return numeric 0');
assertEqual(parseNullableNumber('-0'), 0, 'String "-0" must return numeric 0');

// Test 4: Valid positive and negative numbers
assertEqual(parseNullableNumber('12.5'), 12.5, 'Positive float must parse correctly');
assertEqual(parseNullableNumber('-7.1884'), -7.1884, 'Negative float must parse correctly');
assertEqual(parseNullableNumber('+42'), 42, 'Explicit plus sign must parse correctly');
assertEqual(parseNullableNumber('1e3'), 1000, 'Scientific notation must parse correctly');

// Test 5: Invalid numeric text returns null
assertEqual(parseNullableNumber('invalid'), null, 'Alphabetic text must return null');
assertEqual(parseNullableNumber('NaN'), null, 'NaN string must return null');
assertEqual(parseNullableNumber('12abc'), null, 'Mixed text must return null');
assertEqual(parseNullableNumber(undefined), null, 'undefined must return null');
assertEqual(parseNullableNumber(null), null, 'null must return null');

console.log('All parseNullableNumber unit tests passed successfully!');
