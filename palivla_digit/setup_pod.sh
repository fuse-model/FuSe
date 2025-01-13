sudo apt -y update && sudo apt -y install nfs-common
sudo mkdir -p -m 777 /nfs/nfs3
sudo mount -o rw,intr 10.105.46.66:/nfs3 /nfs/nfs3

sudo usermod -u 3210 kstachowicz
sudo groupmod -g 3210 kstachowicz
sudo chown -R kstachowicz:kstachowicz /home/kstachowicz

sudo -i -u kstachowicz bash << EOF

git config --global --add safe.directory '*'

rm -f .bashrc
ln -s /nfs/nfs3/users/kstachowicz/.bashrc .bashrc
ln -s /nfs/nfs3/users/kstachowicz/.netrc .netrc
EOF

/nfs/nfs3/users/kstachowicz/miniforge3/bin/conda init bash