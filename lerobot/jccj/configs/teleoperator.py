from lerobot.common.teleoperators.so100_leader import SO100Leader, SO100LeaderConfig

teleoperator_config = SO100LeaderConfig(
    port="/dev/tty.usbmodem58FD0171191",
    id="leader",
)
