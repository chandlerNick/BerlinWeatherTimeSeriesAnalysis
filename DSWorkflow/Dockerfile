# Use Python 3.11.9 as the base image
FROM python:3.11.9

# Update the package list and upgrade existing packages to their latest versions
# Install additional dependencies required for the container:
# - sudo: Allows admin privilege
# - rsync: For file synchronization
# - openssh-server: To enable SSH access
# - ffmpeg, libsm6, libxext6: Libraries for multimedia processing
# - htop: A process monitoring tool
RUN apt-get update && apt-get upgrade -y && apt-get install -y sudo rsync openssh-server ffmpeg libsm6 libxext6 htop

# Configure the SSH server:
# - Create the directory for the SSH daemon to store its runtime files.
# - Create the root user's `.ssh` directory and set appropriate permissions.
# - Create an empty `authorized_keys` file where public keys for SSH authentication will be stored.
RUN mkdir -p /var/run/sshd; \
    mkdir /root/.ssh && chmod 700 /root/.ssh; \
    touch /root/.ssh/authorized_keys

# Copy the SSH server configuration file (sshd_config) from the build context into the container.
# This file configures how the SSH server behaves (which port to use, authentication methods etc.).
COPY sshd_config /etc/ssh/sshd_config

# Install the `screen` package, which keeps processes running even if the connection to the terminal is closed.
RUN apt-get install -y screen

# Set the working directory inside the container.
# This is where the application code or other files will be copied and executed.
WORKDIR /storage/courses

# (Optional) COPY <src-path> <destination-path>
# Copy all files from the <src-path> directory on your local machine to <destination-path> inside the Docker image.
# This is useful for including application code or scripts in the container that are not version controlled.
#COPY src /storage/courses/src

# Expose port 22 to allow SSH connections to the container.
EXPOSE 22

# Set the default command to run when the container starts:
# - `/usr/sbin/sshd`: Starts the OpenSSH server.
# - `-D`: Keeps the SSH server running in the foreground (prevents the container from exiting immediately).
CMD ["/usr/sbin/sshd", "-D"]
