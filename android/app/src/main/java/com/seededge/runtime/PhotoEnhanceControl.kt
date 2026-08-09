package com.seededge.runtime

/** Pure, testable command/state contract shared by the foreground service and UI. */
object PhotoEnhanceControl {
    enum class State { IDLE, RUNNING, PAUSE_REQUESTED, PAUSED, CANCEL_REQUESTED, CANCELLED, COMPLETED, FAILED }
    enum class Command { START, PAUSE, PAUSE_ACK, RESUME, CANCEL, CANCEL_ACK, COMPLETE, FAIL }

    fun transition(state: State, command: Command): State = when (state to command) {
        State.IDLE to Command.START -> State.RUNNING
        State.CANCELLED to Command.START,
        State.COMPLETED to Command.START,
        State.FAILED to Command.START -> State.RUNNING
        State.RUNNING to Command.PAUSE -> State.PAUSE_REQUESTED
        State.PAUSE_REQUESTED to Command.PAUSE_ACK -> State.PAUSED
        State.PAUSED to Command.RESUME -> State.RUNNING
        State.RUNNING to Command.CANCEL,
        State.PAUSE_REQUESTED to Command.CANCEL,
        State.PAUSED to Command.CANCEL -> State.CANCEL_REQUESTED
        State.CANCEL_REQUESTED to Command.CANCEL_ACK -> State.CANCELLED
        State.RUNNING to Command.COMPLETE -> State.COMPLETED
        State.RUNNING to Command.FAIL,
        State.PAUSE_REQUESTED to Command.FAIL,
        State.CANCEL_REQUESTED to Command.FAIL -> State.FAILED
        else -> throw IllegalStateException("invalid photo task transition: $state + $command")
    }
}
