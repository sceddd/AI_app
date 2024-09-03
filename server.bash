while getopts ":b" opt; do
  case $opt in
    b)
      python3 manage.py download_weight
      python3 manage.py makemigrations
      python3 manage.py migrate
    ;;

    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

python3 manage.py runserver