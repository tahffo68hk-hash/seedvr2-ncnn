package com.seededge.runtime

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class SafeResultTextTest {
    @Test fun replacesLongWithShortWithoutRetainingRanges() { assertEquals("x".repeat(22), SafeResultText.replace("x".repeat(228), "x".repeat(22))) }
    @Test fun replacesShortWithLong() { assertEquals(228, SafeResultText.replace("x".repeat(22), "x".repeat(228)).length) }
    @Test fun acceptsEmpty() { assertEquals("", SafeResultText.replace("old", "")) }
    @Test fun trimsLongJson() { assertEquals(16_000, SafeResultText.normalize("{" + "x".repeat(20_000) + "}").length) }
    @Test fun preservesUnicodeEmojiAndLines() { assertEquals("中文🙂\nline", SafeResultText.normalize("中文🙂\nline")) }
    @Test fun stripsNullCharacters() { assertEquals("ab", SafeResultText.normalize("a\u0000b")) }
    @Test fun repeatedUpdatesRemainBounded() { var text=""; repeat(100){ text=SafeResultText.append(text,"entry-$it") }; assertTrue(text.length<=16_000); assertTrue(text.endsWith("entry-99\n")) }
    @Test fun activityRebuildUsesPlainValueOnly() { val before=SafeResultText.append("","first"); val rebuilt=SafeResultText.replace("",before); assertEquals(before,rebuilt) }
    @Test fun resultIsImmutableStringNotSpannedState() { val value:Any=SafeResultText.replace("x".repeat(228),"x".repeat(22)); assertTrue(value is String); assertFalse(value is java.lang.StringBuilder) }
}
