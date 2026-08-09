package com.seededge.runtime

import org.junit.Assert.assertEquals
import org.junit.Test

class PhotoEnhanceControlTest {
    @Test
    fun pauseResumeAndCancelAreDistinctBoundaryStates() {
        var state = PhotoEnhanceControl.State.IDLE
        state = PhotoEnhanceControl.transition(state, PhotoEnhanceControl.Command.START)
        state = PhotoEnhanceControl.transition(state, PhotoEnhanceControl.Command.PAUSE)
        assertEquals(PhotoEnhanceControl.State.PAUSE_REQUESTED, state)
        state = PhotoEnhanceControl.transition(state, PhotoEnhanceControl.Command.PAUSE_ACK)
        assertEquals(PhotoEnhanceControl.State.PAUSED, state)
        state = PhotoEnhanceControl.transition(state, PhotoEnhanceControl.Command.RESUME)
        assertEquals(PhotoEnhanceControl.State.RUNNING, state)
        state = PhotoEnhanceControl.transition(state, PhotoEnhanceControl.Command.CANCEL)
        assertEquals(PhotoEnhanceControl.State.CANCEL_REQUESTED, state)
        state = PhotoEnhanceControl.transition(state, PhotoEnhanceControl.Command.CANCEL_ACK)
        assertEquals(PhotoEnhanceControl.State.CANCELLED, state)
    }

    @Test(expected = IllegalStateException::class)
    fun cancelledTaskCannotResume() {
        PhotoEnhanceControl.transition(PhotoEnhanceControl.State.CANCELLED,
            PhotoEnhanceControl.Command.RESUME)
    }

    @Test
    fun terminalTaskAllowsASeparateNewTask() {
        assertEquals(PhotoEnhanceControl.State.RUNNING,
            PhotoEnhanceControl.transition(PhotoEnhanceControl.State.COMPLETED,
                PhotoEnhanceControl.Command.START))
        assertEquals(PhotoEnhanceControl.State.RUNNING,
            PhotoEnhanceControl.transition(PhotoEnhanceControl.State.FAILED,
                PhotoEnhanceControl.Command.START))
    }
}
